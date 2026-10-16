from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .confidence import detect_handoff, score_confidence
from .config import Settings
from .crm_client import ToolContext
from .latency import MetricsStore, StageTimer, TurnMetrics
from .prompt import build_system_prompt
from .rag import Hit, RAGEngine
from .sessions import CallSession, ToolRecord, TurnRecord
from .tools import TOOL_NAMES, TOOL_SPECS, ToolExecutor
from .tts import TTSAudio, synthesize_long

logger = logging.getLogger("voice.agent")

LLM_FAILURE_REPLY = (
    "I am having trouble reaching my systems right now. "
    "Let me connect you with a specialist who can help."
)


@dataclass
class Thought:
    reply: str
    confidence: float
    handoff: bool
    handoff_reason: str
    session: CallSession
    metrics: TurnMetrics
    hits: list[Hit] = field(default_factory=list)
    tool_records: list[ToolRecord] = field(default_factory=list)
    user_text: str = ""
    ok: bool = True
    error: str | None = None
    committed: bool = False


@dataclass
class TurnOutcome:
    thought: Thought
    audio: bytes = b""
    tts: TTSAudio | None = None

    @property
    def reply(self) -> str:
        return self.thought.reply

    @property
    def confidence(self) -> float:
        return self.thought.confidence

    @property
    def handoff(self) -> bool:
        return self.thought.handoff

    @property
    def metrics(self) -> TurnMetrics:
        return self.thought.metrics


class VoiceAgent:
    def __init__(
        self,
        settings: Settings,
        rag: RAGEngine,
        llm: Any,
        stt: Any,
        tts: Any,
        tools: ToolExecutor,
        metrics: MetricsStore,
    ) -> None:
        self.settings = settings
        self.rag = rag
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.tools = tools
        self.metrics = metrics

    # ------------------------------------------------------------------ tools
    def tool_context(self, session: CallSession) -> ToolContext:
        return ToolContext(
            caller_id=session.caller_id,
            call_sid=session.call_sid,
            session_id=session.session_id,
            called=session.called,
        )

    def _executor(self, session: CallSession):
        async def executor(name: str, arguments: dict[str, Any]) -> tuple[str, ToolRecord]:
            ctx = self.tool_context(session)
            content, record = await self.tools.execute(name, arguments, ctx)
            session.tool_records.append(record)
            return content, record

        return executor

    # -------------------------------------------------------------- pipeline
    async def think(
        self,
        text: str,
        session: CallSession,
        *,
        vad_ms: float = 0.0,
        stt_ms: float = 0.0,
        stt_confidence: float = 1.0,
    ) -> Thought:
        turn_index = session.turn_count + 1
        entry = time.perf_counter()
        metrics = TurnMetrics(
            turn_index=turn_index,
            call_sid=session.call_sid,
            session_id=session.session_id,
            caller_id=session.caller_id,
            vad_ms=vad_ms,
            stt_ms=stt_ms,
            user_chars=len(text),
            model=getattr(self.llm, "model", getattr(self.llm, "name", "")),
            stt_provider=getattr(self.stt, "name", ""),
        )

        handoff_reason = detect_handoff(text)
        if handoff_reason:
            thought = Thought(
                reply=self.settings.handoff_ack,
                confidence=0.0,
                handoff=True,
                handoff_reason=handoff_reason,
                session=session,
                metrics=metrics,
                user_text=text,
            )
            return thought

        with StageTimer() as retrieve_timer:
            hits = self.rag.retrieve(text)
        metrics.retrieve_ms = retrieve_timer.elapsed_ms
        context = self.rag.build_context(hits)

        system = build_system_prompt(
            self.settings,
            context,
            caller_id=session.caller_id,
            call_sid=session.call_sid,
            tool_names=TOOL_NAMES,
        )
        messages: list[dict[str, Any]] = [ {"role": "system", "content": system} ]
        messages.extend(session.messages)
        messages.append({"role": "user", "content": text})

        llm_result = await self.llm.generate(
            messages, tools=TOOL_SPECS, executor=self._executor(session)
        )
        metrics.llm_ms = llm_result.latency_ms
        metrics.tool_ms = llm_result.tool_ms
        metrics.tool_calls = len(llm_result.tool_calls)
        tool_records = list(llm_result.tool_calls)

        reply = llm_result.reply.strip()
        if not llm_result.ok or not reply:
            reply = LLM_FAILURE_REPLY
            handoff_reason = handoff_reason or "llm_error"
            metrics.model = metrics.model or llm_result.model

        intent = handoff_reason or ("llm:handoff_marker" if llm_result.handoff_marker else None)
        confidence = score_confidence(
            utterance=text,
            reply=reply,
            hits=hits,
            provider=self.rag.provider,
            min_score=self.rag.min_score,
            tool_records=tool_records,
            handoff_intent=intent,
            stt_confidence=stt_confidence,
        )
        handoff = intent is not None or confidence < self.settings.confidence_threshold
        reason = intent or ("low_confidence" if handoff else "")

        thought = Thought(
            reply=reply,
            confidence=confidence,
            handoff=handoff,
            handoff_reason=reason,
            session=session,
            metrics=metrics,
            hits=hits,
            tool_records=tool_records,
            user_text=text,
            ok=llm_result.ok,
            error=llm_result.error,
        )
        metrics.confidence = confidence
        metrics.handoff = handoff
        metrics.total_ms = (time.perf_counter() - entry) * 1000.0 + vad_ms
        return thought

    def repeat_thought(
        self,
        session: CallSession,
        *,
        vad_ms: float = 0.0,
        stt_ms: float = 0.0,
        reason: str = "stt_empty",
    ) -> Thought:
        """Low-latency fallback when the caller's audio could not be transcribed."""
        metrics = TurnMetrics(
            turn_index=session.turn_count + 1,
            call_sid=session.call_sid,
            session_id=session.session_id,
            caller_id=session.caller_id,
            vad_ms=vad_ms,
            stt_ms=stt_ms,
            confidence=0.5,
            model=getattr(self.llm, "name", ""),
            stt_provider=getattr(self.stt, "name", ""),
        )
        return Thought(
            reply=self.settings.repeat_prompt,
            confidence=0.5,
            handoff=False,
            handoff_reason="",
            session=session,
            metrics=metrics,
            user_text="",
            error=reason,
        )

    async def speak(self, thought: Thought) -> TTSAudio:
        if not thought.reply:
            return TTSAudio(mulaw=b"", provider="none", model="none")
        return await synthesize_long(self.tts, thought.reply, self.settings.tts_max_chars)

    async def speak_text(self, text: str) -> TTSAudio:
        return await synthesize_long(self.tts, text, self.settings.tts_max_chars)

    def commit(
        self,
        thought: Thought,
        *,
        tts_ms: float = 0.0,
        vad_ms: float = 0.0,
        interrupted: bool = False,
        synthesize: bool = True,
    ) -> TurnMetrics:
        """Persist history, transcript and metrics for a finished turn."""
        if thought.committed:
            return thought.metrics
        thought.committed = True
        session = thought.session
        metrics = thought.metrics
        metrics.vad_ms = max(metrics.vad_ms, vad_ms)
        metrics.tts_ms = tts_ms
        metrics.confidence = thought.confidence
        metrics.handoff = thought.handoff
        metrics.interrupted = interrupted
        metrics.reply_chars = len(thought.reply)
        if synthesize:
            metrics.total_ms = metrics.stage_sum_ms + metrics.vad_ms

        if thought.user_text:
            session.messages.append({"role": "user", "content": thought.user_text})
        if thought.reply and not interrupted:
            session.messages.append({"role": "assistant", "content": thought.reply})
        overflow = len(session.messages) - self.settings.max_history_messages
        if overflow > 0:
            del session.messages[:overflow]

        session.turns.append(
            TurnRecord(
                index=metrics.turn_index,
                user_text=thought.user_text,
                reply=thought.reply,
                confidence=thought.confidence,
                handoff=thought.handoff,
                latency_ms=metrics.total_ms,
            )
        )
        if thought.handoff and not session.transfer_requested:
            session.transfer_requested = True
            session.transfer_reason = thought.handoff_reason or "agent decision"
        session.touch()
        self.metrics.record(metrics)
        return metrics

    async def handle_audio_turn(
        self,
        ulaw: bytes,
        session: CallSession,
        *,
        vad_ms: float = 0.0,
    ) -> TurnOutcome:
        entry = time.perf_counter()
        stt_result = await self.stt.transcribe(ulaw)
        stt_ms = stt_result.duration_ms
        no_speech = stt_result.no_speech_prob > self.settings.stt_no_speech_max
        if not stt_result.text.strip() or no_speech:
            thought = self.repeat_thought(session, vad_ms=vad_ms, stt_ms=stt_ms)
        else:
            thought = await self.think(
                stt_result.text,
                session,
                vad_ms=vad_ms,
                stt_ms=stt_ms,
                stt_confidence=stt_result.confidence,
            )
        tts_audio = await self.speak(thought)
        self.commit(thought, tts_ms=tts_audio.latency_ms, vad_ms=vad_ms)
        thought.metrics.total_ms = (time.perf_counter() - entry) * 1000.0 + vad_ms
        return TurnOutcome(thought=thought, audio=tts_audio.mulaw, tts=tts_audio)

    async def handle_text_turn(
        self,
        text: str,
        session: CallSession,
        *,
        synthesize: bool = True,
        vad_ms: float = 0.0,
        stt_ms: float = 0.0,
        stt_confidence: float = 1.0,
    ) -> TurnOutcome:
        entry = time.perf_counter()
        thought = await self.think(
            text,
            session,
            vad_ms=vad_ms,
            stt_ms=stt_ms,
            stt_confidence=stt_confidence,
        )
        tts_audio: TTSAudio | None = None
        audio = b""
        if synthesize:
            tts_audio = await self.speak(thought)
            audio = tts_audio.mulaw
        self.commit(
            thought,
            tts_ms=tts_audio.latency_ms if tts_audio else 0.0,
            vad_ms=vad_ms,
            synthesize=synthesize,
        )
        thought.metrics.total_ms = (time.perf_counter() - entry) * 1000.0 + vad_ms
        return TurnOutcome(thought=thought, audio=audio, tts=tts_audio)

    async def handle_text(
        self,
        text: str,
        session: CallSession,
        *,
        synthesize: bool = False,
    ) -> TurnOutcome:
        return await self.handle_text_turn(text, session, synthesize=synthesize)
