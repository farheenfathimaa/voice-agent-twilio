from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import audio
from .config import Settings

logger = logging.getLogger("voice.tts")

GROQ_SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"
OPENAI_SPEECH_URL = "https://api.openai.com/v1/audio/speech"


@dataclass
class TTSAudio:
    mulaw: bytes
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    chars: int = 0
    used_fallback: bool = False
    error: str | None = None

    @property
    def duration_ms(self) -> int:
        return audio.ulaw_duration_ms(self.mulaw)


def split_for_tts(text: str, max_chars: int = 400) -> list[str]:
    """Split on sentence boundaries, merging short sentences up to max_chars."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            piece, sentence = sentence[:max_chars], sentence[max_chars:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


class _BaseTTS:
    name = "base"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.tts_timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def _speech(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> bytes:
        resp = await self._client.post(
            url, headers=headers, json=body, timeout=self.settings.tts_timeout_s
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"tts HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.content


def _to_mulaw(payload: bytes) -> bytes:
    if payload.startswith(b"RIFF"):
        return audio.audio_to_ulaw8k(payload)
    raise RuntimeError("tts response was not wav/RIFF audio")


class GroqTTS(_BaseTTS):
    name = "groq"

    async def synthesize(self, text: str) -> TTSAudio:
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.settings.groq_api_key}"}
        body: dict[str, Any] = {
            "model": self.settings.tts_model,
            "voice": self.settings.tts_voice,
            "input": text,
            "response_format": "wav",
            "sample_rate": audio.TWILIO_RATE,
        }
        try:
            payload = await self._speech(GROQ_SPEECH_URL, headers, body)
        except Exception:
            body.pop("sample_rate", None)
            try:
                payload = await self._speech(GROQ_SPEECH_URL, headers, body)
            except Exception as exc:  # noqa: BLE001
                if not self.settings.tts_fallback_tone:
                    raise
                return self._tone_fallback(text, started, str(exc))
        try:
            mulaw = _to_mulaw(payload)
        except Exception as exc:  # noqa: BLE001
            if not self.settings.tts_fallback_tone:
                raise
            return self._tone_fallback(text, started, str(exc))
        return TTSAudio(
            mulaw=mulaw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.name,
            model=self.settings.tts_model,
            chars=len(text),
        )

    def _tone_fallback(self, text: str, started: float, error: str) -> TTSAudio:
        return tone_fallback(text, started, error)


def tone_fallback(text: str, started: float, error: str) -> TTSAudio:
    logger.error("tts failed, using tone fallback: %s", error)
    duration = audio.speech_duration_ms_for_text(text)
    return TTSAudio(
        mulaw=audio.tone_ulaw(duration),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        provider="tone",
        model="tone",
        chars=len(text),
        used_fallback=True,
        error=error,
    )


class OpenAITTS(_BaseTTS):
    name = "openai"

    async def synthesize(self, text: str) -> TTSAudio:
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        body = {
            "model": self.settings.tts_model,
            "voice": self.settings.tts_voice,
            "input": text,
            "response_format": "pcm",
        }
        try:
            payload = await self._speech(OPENAI_SPEECH_URL, headers, body)
            pcm = audio.resample_pcm16(payload, 24000, audio.TWILIO_RATE)
            mulaw = audio.pcm16_to_ulaw(pcm)
        except Exception as exc:  # noqa: BLE001
            if not self.settings.tts_fallback_tone:
                raise
            return tone_fallback(text, started, str(exc))
        return TTSAudio(
            mulaw=mulaw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.name,
            model=self.settings.tts_model,
            chars=len(text),
        )


class MockTTS:
    """Deterministic TTS: tone sized to the text length. Records every request."""

    name = "mock"
    model = "mock-tts"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> TTSAudio:
        self.calls.append(text)
        started = time.perf_counter()
        duration = audio.speech_duration_ms_for_text(text)
        return TTSAudio(
            mulaw=audio.tone_ulaw(duration, freq=420.0),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.name,
            model=self.model,
            chars=len(text),
        )

    async def close(self) -> None:
        return None


def build_tts(settings: Settings, client: httpx.AsyncClient | None = None) -> Any:
    provider = settings.tts_provider.lower()
    if provider == "groq":
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY missing; falling back to mock TTS")
            return MockTTS(settings, client)
        return GroqTTS(settings, client)
    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY missing; falling back to mock TTS")
            return MockTTS(settings, client)
        return OpenAITTS(settings, client)
    if provider == "mock":
        return MockTTS(settings, client)
    raise ValueError(f"unknown tts provider: {provider}")


async def synthesize_long(tts: Any, text: str, max_chars: int) -> TTSAudio:
    """Synthesize a reply in sentence-sized chunks and concatenate mu-law frames."""
    chunks = split_for_tts(text, max_chars) or [text]
    total_started = time.perf_counter()
    frames: list[bytes] = []
    provider = ""
    model = ""
    fallback = False
    error: str | None = None
    for chunk in chunks:
        audio_part = await tts.synthesize(chunk)
        frames.append(audio_part.mulaw)
        provider = provider or audio_part.provider
        model = model or audio_part.model
        fallback = fallback or audio_part.used_fallback
        error = error or audio_part.error
    return TTSAudio(
        mulaw=b"".join(frames),
        latency_ms=(time.perf_counter() - total_started) * 1000.0,
        provider=provider,
        model=model,
        chars=len(text),
        used_fallback=fallback,
        error=error,
    )
