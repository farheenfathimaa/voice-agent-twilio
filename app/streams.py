"""Bidirectional Twilio Media Streams pipeline: VAD -> STT -> LLM/RAG -> TTS."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from . import audio
from .bootstrap import Components
from .config import Settings
from .sessions import CallSession

logger = logging.getLogger("voice.stream")

IDLE = "idle"
SPEAKING = "speaking"
THINKING = "thinking"


class Endpointer:
    """Energy-based voice activity detection over 20 ms mu-law frames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.frames: list[bytes] = []
        self.speaking = False
        self.speech_started_at = 0.0
        self.last_voice_at = 0.0

    def reset(self) -> None:
        self.frames = []
        self.speaking = False
        self.speech_started_at = 0.0
        self.last_voice_at = 0.0

    def feed(self, payload: bytes) -> list[str]:
        settings = self.settings
        now = time.monotonic()
        energy = audio.ulaw_rms(payload)
        events: list[str] = []
        if not self.speaking:
            if energy >= settings.vad_start_rms:
                self.speaking = True
                self.frames = [payload]
                self.speech_started_at = now
                self.last_voice_at = now
                events.append("speech_start")
            else:
                self.frames = []
            return events

        self.frames.append(payload)
        if energy >= settings.vad_end_rms:
            self.last_voice_at = now
        speech_ms = (now - self.speech_started_at) * 1000.0
        silence_ms = (now - self.last_voice_at) * 1000.0
        if silence_ms >= settings.vad_silence_ms or speech_ms >= settings.max_utterance_s * 1000:
            events.append("endpoint")

        max_frames = int(settings.max_buffer_s * 1000 / settings.frame_ms)
        if len(self.frames) > max_frames:
            self.frames = self.frames[-max_frames:]
        return events

    def take(self) -> bytes:
        frames = b"".join(self.frames)
        duration_ms = audio.ulaw_duration_ms(frames)
        self.reset()
        if duration_ms < self.settings.vad_min_speech_ms:
            return b""
        return frames


class MediaConnection:
    def __init__(self, websocket: WebSocket, settings: Settings, components: Components) -> None:
        self.ws = websocket
        self.settings = settings
        self.c = components
        self.sessions = components.sessions
        self.session: CallSession | None = None
        self.stream_sid = ""
        self.state = IDLE
        self.endpointer = Endpointer(settings)
        self.send_lock = asyncio.Lock()
        self.turn_task: asyncio.Task[Any] | None = None
        self.mark_events: dict[str, asyncio.Event] = {}
        self.mark_counter = 0
        self.current_mark: asyncio.Event | None = None
        self.interrupted = False
        self.barge_frames = 0
        self.pending_audio = b""
        self.closed = False
        self.transferring = False
        self.stop_received = False
        self.stop_reason = ""

    # ------------------------------------------------------------------ loop
    async def run(self) -> None:
        try:
            await self.ws.accept()
            while True:
                message = await self.ws.receive()
                mtype = message.get("type")
                if mtype == "websocket.disconnect":
                    break
                if mtype != "websocket.receive":
                    break
                text = message.get("text")
                if not text:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("non-json frame dropped")
                    continue
                await self._dispatch(data)
                if self.stop_received:
                    break
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("media stream crashed (stream=%s)", self.stream_sid)
        finally:
            await self._finalize()

    async def _dispatch(self, data: dict[str, Any]) -> None:
        event = data.get("event")
        if event == "connected":
            logger.debug("media stream connected")
        elif event == "start":
            self._on_start(data)
        elif event == "media":
            await self._on_media(data)
        elif event == "mark":
            self._on_mark(data)
        elif event == "dtmf":
            await self._on_dtmf(data)
        elif event == "stop":
            self.stop_received = True
            self.stop_reason = "twilio_stop"

    # --------------------------------------------------------------- events
    def _on_start(self, data: dict[str, Any]) -> None:
        start = data.get("start") or {}
        custom = start.get("customParameters") or {}
        self.stream_sid = data.get("streamSid") or start.get("streamSid") or ""
        call_sid = start.get("callSid") or custom.get("call_sid") or ""
        caller_id = custom.get("caller_id") or custom.get("From") or ""
        called = custom.get("called") or custom.get("To") or ""
        self.session = self.sessions.get_or_create(
            call_sid=call_sid, caller_id=caller_id, called=called, source="stream"
        )
        self.session.stream_sid = self.stream_sid
        self.session.active = True
        self.session.ended = False
        self.state = IDLE
        logger.info(
            "stream start call_sid=%s caller=%s stream=%s",
            self.session.call_sid,
            caller_id,
            self.stream_sid,
        )
        if self.settings.greeting:
            self.turn_task = asyncio.create_task(self._speak_greeting())

    async def _on_media(self, data: dict[str, Any]) -> None:
        if self.closed or self.transferring:
            return
        media = data.get("media") or {}
        payload_b64 = media.get("payload")
        if not payload_b64:
            return
        try:
            payload = base64.b64decode(payload_b64)
        except (ValueError, TypeError):
            return

        if self.state == SPEAKING:
            energy = audio.ulaw_rms(payload)
            if energy >= self.settings.vad_start_rms:
                self.barge_frames += 1
            else:
                self.barge_frames = 0
            if self.barge_frames >= self.settings.vad_barge_frames:
                logger.debug("barge-in detected during playback")
                await self._interrupt_playback()
                self.state = IDLE
                self.endpointer.reset()
            return

        events = self.endpointer.feed(payload)
        if "speech_start" in events and self.state == IDLE:
            logger.debug("speech started")
        if "endpoint" not in events:
            return

        speech_started = self.endpointer.speech_started_at
        utterance = self.endpointer.take()
        if not utterance:
            return
        vad_ms = (time.monotonic() - speech_started) * 1000.0 if speech_started else 0.0

        if self.state == THINKING:
            self.pending_audio = utterance
            logger.debug("utterance queued while thinking (%d bytes)", len(utterance))
            return
        if self.state == IDLE:
            self._start_turn(utterance, vad_ms)

    def _on_mark(self, data: dict[str, Any]) -> None:
        name = (data.get("mark") or {}).get("name")
        if not name:
            return
        event = self.mark_events.pop(name, None)
        if event is not None:
            event.set()

    async def _on_dtmf(self, data: dict[str, Any]) -> None:
        digit = str((data.get("dtmf") or {}).get("digit") or "")
        if not digit or self.transferring or self.session is None:
            return
        if digit != self.settings.handoff_digit:
            return
        logger.info("dtmf handoff request digit=%s call=%s", digit, self.session.call_sid)
        self.session.transfer_requested = True
        self.session.transfer_reason = f"dtmf {digit}"
        self.transferring = True
        await self._interrupt_playback()
        self.turn_task = asyncio.create_task(self._transfer_with_ack())

    # ----------------------------------------------------------- turn flow
    def _start_turn(self, utterance: bytes, vad_ms: float = 0.0) -> None:
        if self.closed or self.session is None:
            return
        if self.turn_task is not None and not self.turn_task.done():
            self.pending_audio = utterance
            return
        self.turn_task = asyncio.create_task(self._run_turn(utterance, vad_ms))

    async def _speak_greeting(self) -> None:
        try:
            audio_part = await self.c.agent.speak_text(self.settings.greeting)
            if audio_part.mulaw:
                await self._play(audio_part.mulaw, "greet")
        except Exception:  # noqa: BLE001
            logger.exception("greeting playback failed")
        finally:
            if self.state == SPEAKING:
                self.state = IDLE

    async def _run_turn(self, utterance: bytes, vad_ms: float) -> None:
        session = self.session
        if session is None:
            return
        self.state = THINKING
        try:
            entry = time.perf_counter()
            stt_result = await self.c.stt.transcribe(utterance)
            if not stt_result.text.strip() or (
                stt_result.no_speech_prob > self.settings.stt_no_speech_max
            ):
                thought = self.c.agent.repeat_thought(
                    session, vad_ms=vad_ms, stt_ms=stt_result.duration_ms
                )
            else:
                thought = await self.c.agent.think(
                    stt_result.text,
                    session,
                    vad_ms=vad_ms,
                    stt_ms=stt_result.duration_ms,
                    stt_confidence=stt_result.confidence,
                )

            if self.closed:
                self.c.agent.commit(thought, vad_ms=vad_ms, synthesize=False, interrupted=True)
                return

            if self.pending_audio:
                nxt, self.pending_audio = self.pending_audio, b""
                self.c.agent.commit(
                    thought, vad_ms=vad_ms, synthesize=False, interrupted=True
                )
                logger.debug("yielding to caller interruption before tts")
                self._start_turn(nxt, vad_ms=0.0)
                return

            audio_part = await self.c.agent.speak(thought)
            completed = await self._play(audio_part.mulaw, "say")
            self.c.agent.commit(
                thought,
                tts_ms=audio_part.latency_ms,
                vad_ms=vad_ms,
                interrupted=not completed,
            )
            logger.debug(
                "turn done in %.0fms confidence=%.2f handoff=%s interrupted=%s",
                (time.perf_counter() - entry) * 1000.0,
                thought.confidence,
                thought.handoff,
                not completed,
            )
            if thought.handoff and session.transfer_requested and not self.transferring:
                self.transferring = True
                await self._finish_transfer()
                return
            if self.state == SPEAKING:
                self.state = IDLE
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("turn failed (call=%s)", session.call_sid)
            if self.state == THINKING:
                self.state = IDLE

    # ------------------------------------------------------------ playback
    async def _send(self, message: dict[str, Any]) -> None:
        async with self.send_lock:
            if self.closed:
                return
            try:
                await self.ws.send_text(json.dumps(message))
            except Exception:  # noqa: BLE001
                self.closed = True

    async def _play(self, ulaw: bytes, tag: str) -> bool:
        """Stream mu-law frames and wait for Twilio's mark. False = interrupted."""
        if not ulaw or self.closed:
            return not self.interrupted
        name = f"{tag}-{self.mark_counter}"
        self.mark_counter += 1
        event = asyncio.Event()
        self.mark_events[name] = event
        self.current_mark = event
        self.interrupted = False
        self.state = SPEAKING
        frames = audio.split_ulaw(ulaw, self.settings.frame_ms)
        for frame in frames:
            if self.interrupted or self.closed:
                break
            await self._send(
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(frame).decode("ascii")},
                }
            )
        await self._send(
            {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": name},
            }
        )
        timeout_s = audio.ulaw_duration_ms(ulaw) / 1000.0 + 15.0
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("mark %s not acknowledged by twilio", name)
        self.current_mark = None
        self.mark_events.pop(name, None)
        return not self.interrupted

    async def _interrupt_playback(self) -> None:
        self.interrupted = True
        event = self.current_mark
        if event is not None:
            event.set()
        await self._send({"event": "clear", "streamSid": self.stream_sid})

    # ------------------------------------------------------------- handoff
    async def _transfer_with_ack(self) -> None:
        try:
            self.state = THINKING
            audio_part = await self.c.agent.speak_text(self.settings.handoff_ack)
            await self._play(audio_part.mulaw, "transfer")
        except Exception:  # noqa: BLE001
            logger.exception("transfer ack failed")
        finally:
            await self._finish_transfer()

    async def _finish_transfer(self) -> None:
        self.transferring = True
        logger.info(
            "closing media stream to trigger <Dial> (call=%s)",
            self.session.call_sid if self.session else "?",
        )
        await asyncio.sleep(0.05)
        self.closed = True
        try:
            await self.ws.close(code=1000)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------- teardown
    async def _finalize(self) -> None:
        self.closed = True
        task = self.turn_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        session = self.session
        if session is None:
            return
        session.active = False
        session.touch()
        await self.sessions.finalize(session, reason=self.stop_reason or "disconnect")
