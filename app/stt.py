from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import audio
from .config import Settings

logger = logging.getLogger("voice.stt")

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"


@dataclass
class STTResult:
    text: str
    confidence: float = 0.0
    no_speech_prob: float = 0.0
    duration_ms: float = 0.0
    audio_ms: float = 0.0
    provider: str = ""
    model: str = ""
    ok: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "no_speech_prob": round(self.no_speech_prob, 3),
            "duration_ms": round(self.duration_ms, 1),
            "audio_ms": round(self.audio_ms, 1),
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
        }


class _BaseSTT:
    name = "base"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.stt_timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(self, url: str, headers: dict[str, str], payload: dict[str, str],
                    wav: bytes) -> httpx.Response:
        files = {"file": ("utterance.wav", wav, "audio/wav")}
        return await self._client.post(
            url, headers=headers, data=payload, files=files, timeout=self.settings.stt_timeout_s
        )


def _confidence_from(payload: dict[str, Any], text: str) -> tuple[float, float]:
    if not text.strip():
        return 0.0, 1.0
    no_speech = float(payload.get("no_speech_prob") or 0.0)
    segments = payload.get("segments") or []
    avg_logprob = payload.get("avg_logprob")
    if avg_logprob is None and segments:
        logs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
        if logs:
            avg_logprob = sum(logs) / len(logs)
    conf = 1.0 - min(max(no_speech, 0.0), 1.0)
    if avg_logprob is not None:
        conf *= min(max(math.exp(float(avg_logprob)), 0.0), 1.0)
    return min(max(conf, 0.0), 1.0), no_speech


def _parse_transcription(resp: httpx.Response, settings: Settings, provider: str,
                          audio_ms: float, started: float) -> STTResult:
    latency = (time.perf_counter() - started) * 1000.0
    if resp.status_code >= 400:
        detail = resp.text[:300]
        logger.error("stt provider %s returned %s: %s", provider, resp.status_code, detail)
        return STTResult(
            text="",
            provider=provider,
            model=settings.stt_model,
            duration_ms=latency,
            audio_ms=audio_ms,
            ok=False,
            error=f"HTTP {resp.status_code}: {detail}",
        )
    payload = resp.json()
    text = str(payload.get("text") or "").strip()
    conf, no_speech = _confidence_from(payload, text)
    return STTResult(
        text=text,
        confidence=conf,
        no_speech_prob=no_speech,
        duration_ms=latency,
        audio_ms=audio_ms,
        provider=provider,
        model=settings.stt_model,
        ok=bool(text) or no_speech < settings.stt_no_speech_max,
        error=None if text else "empty transcript",
    )


class GroqSTT(_BaseSTT):
    name = "groq"

    async def transcribe(self, ulaw: bytes) -> STTResult:
        started = time.perf_counter()
        audio_ms = audio.ulaw_duration_ms(ulaw)
        pcm = audio.ulaw_to_pcm16(ulaw)
        wav = audio.build_wav(pcm, audio.TWILIO_RATE)
        headers = {"Authorization": f"Bearer {self.settings.groq_api_key}"}
        payload = {
            "model": self.settings.stt_model,
            "response_format": "verbose_json",
            "language": self.settings.stt_language,
            "temperature": "0",
        }
        try:
            resp = await self._post(GROQ_TRANSCRIPTIONS_URL, headers, payload, wav)
        except httpx.HTTPError as exc:
            logger.error("stt request failed: %s", exc)
            return STTResult(
                text="",
                provider=self.name,
                model=self.settings.stt_model,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                audio_ms=audio_ms,
                ok=False,
                error=str(exc),
            )
        return _parse_transcription(resp, self.settings, self.name, audio_ms, started)


class OpenAISTT(_BaseSTT):
    name = "openai"

    async def transcribe(self, ulaw: bytes) -> STTResult:
        started = time.perf_counter()
        audio_ms = audio.ulaw_duration_ms(ulaw)
        pcm = audio.ulaw_to_pcm16(ulaw)
        wav = audio.build_wav(pcm, audio.TWILIO_RATE)
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.stt_model,
            "response_format": "verbose_json",
            "language": self.settings.stt_language,
            "temperature": "0",
        }
        try:
            resp = await self._post(OPENAI_TRANSCRIPTIONS_URL, headers, payload, wav)
        except httpx.HTTPError as exc:
            return STTResult(
                text="",
                provider=self.name,
                model=self.settings.stt_model,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                audio_ms=audio_ms,
                ok=False,
                error=str(exc),
            )
        return _parse_transcription(resp, self.settings, self.name, audio_ms, started)


class MockSTT:
    """Deterministic STT for tests / offline eval: pops pre-loaded utterances."""

    name = "mock"
    model = "mock-stt"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.queue: deque[STTResult] = deque()
        self.calls: list[float] = []

    def push(self, text: str, confidence: float = 0.93, no_speech_prob: float = 0.05) -> None:
        audio_ms = audio.speech_duration_ms_for_text(text)
        self.queue.append(
            STTResult(
                text=text,
                confidence=confidence,
                no_speech_prob=no_speech_prob,
                audio_ms=float(audio_ms),
                provider=self.name,
                model=self.model,
            )
        )

    async def transcribe(self, ulaw: bytes) -> STTResult:
        audio_ms = audio.ulaw_duration_ms(ulaw)
        self.calls.append(audio_ms)
        if not self.queue:
            return STTResult(
                text="",
                confidence=0.0,
                no_speech_prob=0.99,
                audio_ms=audio_ms,
                provider=self.name,
                model=self.model,
                ok=False,
                error="empty transcript",
            )
        result = self.queue.popleft()
        result.audio_ms = audio_ms
        result.duration_ms = 1.0
        return result

    async def close(self) -> None:
        return None


def build_stt(settings: Settings, client: httpx.AsyncClient | None = None) -> Any:
    provider = settings.stt_provider.lower()
    if provider == "groq":
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY missing; falling back to mock STT")
            return MockSTT(settings, client)
        return GroqSTT(settings, client)
    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY missing; falling back to mock STT")
            return MockSTT(settings, client)
        return OpenAISTT(settings, client)
    if provider == "mock":
        return MockSTT(settings, client)
    raise ValueError(f"unknown stt provider: {provider}")


@dataclass
class STTState:
    last_result: STTResult | None = field(default=None)
