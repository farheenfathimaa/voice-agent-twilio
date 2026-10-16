"""G.711 mu-law audio helpers for Twilio Media Streams (audio/x-mulaw @ 8 kHz).

Pure numpy implementation (no ffmpeg / audioop dependency).
"""

from __future__ import annotations

import io
import math
import wave

import numpy as np

TWILIO_RATE = 8000
FRAME_MS = 20
FRAME_SAMPLES = TWILIO_RATE * FRAME_MS // 1000  # 160 samples = 160 bytes of mulaw


def _decode_ulaw_byte(u: int) -> int:
    uval = (~u) & 0xFF
    exponent = (uval >> 4) & 0x07
    mantissa = uval & 0x0F
    t = ((mantissa << 3) + 0x84) << exponent
    if uval & 0x80:
        return 0x84 - t
    return t - 0x84


_ULAW_DECODE = np.array([_decode_ulaw_byte(i) for i in range(256)], dtype=np.int16)

_ULAW_LEVELS = _ULAW_DECODE.astype(np.int32)
_ORDER = np.argsort(_ULAW_LEVELS, kind="stable")
_SORTED_LEVELS = _ULAW_LEVELS[_ORDER]


def ulaw_to_pcm16(ulaw: bytes | bytearray | memoryview) -> bytes:
    """Decode mu-law bytes to signed 16-bit little-endian PCM."""
    arr = np.frombuffer(bytes(ulaw), dtype=np.uint8)
    return _ULAW_DECODE[arr].astype("<i2").tobytes()


def pcm16_to_ulaw(pcm: bytes | bytearray | memoryview) -> bytes:
    """Encode signed 16-bit little-endian PCM to mu-law bytes."""
    samples = np.frombuffer(bytes(pcm), dtype="<i2").astype(np.int32)
    if samples.size == 0:
        return b""
    idx = np.searchsorted(_SORTED_LEVELS, samples, side="left")
    idx = np.clip(idx, 0, _SORTED_LEVELS.size - 1)
    left = np.clip(idx - 1, 0, _SORTED_LEVELS.size - 1)
    right = idx
    pick_left = np.abs(_SORTED_LEVELS[left] - samples) <= np.abs(_SORTED_LEVELS[right] - samples)
    chosen = np.where(pick_left, left, right)
    codes = _ORDER[chosen].astype(np.uint8)
    return codes.tobytes()


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono s16le PCM. Integer decimation uses mean-pooling (cheap anti-alias)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return b""
    ratio = src_rate / dst_rate
    if ratio > 1 and abs(ratio - round(ratio)) < 1e-9:
        factor = int(round(ratio))
        usable = (samples.size // factor) * factor
        if usable == 0:
            return b""
        pooled = samples[:usable].reshape(-1, factor).mean(axis=1)
        out = pooled
    else:
        n_out = max(1, int(round(samples.size / ratio)))
        positions = np.arange(samples.size, dtype=np.float64)
        out = np.interp(np.linspace(0, samples.size - 1, n_out), positions, samples)
    clipped = np.clip(np.round(out), -32768, 32767).astype("<i2")
    return clipped.tobytes()


def build_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(data: bytes) -> tuple[bytes, int]:
    """Parse a RIFF/WAVE buffer -> (s16le mono pcm, sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        pcm = frames
    elif width == 1:
        uns = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        pcm = ((uns - 128) << 8).astype("<i2").tobytes()
    elif width == 4:
        ints = np.frombuffer(frames, dtype="<i4").astype(np.int64) >> 16
        pcm = ints.astype("<i2").tobytes()
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        val = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        val = np.where(val & 0x800000, val - 0x1000000, val)
        pcm = (val >> 8).astype("<i2").tobytes()
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if channels > 1:
        mono = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).mean(axis=1)
        pcm = np.clip(np.round(mono), -32768, 32767).astype("<i2").tobytes()
    return pcm, rate


def audio_to_ulaw8k(data: bytes, source_rate: int | None = None) -> bytes:
    """Convert wav bytes (or raw pcm when source_rate given) to 8 kHz mu-law."""
    if source_rate is None:
        if not data.startswith(b"RIFF"):
            raise ValueError("expected RIFF/wav payload")
        pcm, rate = wav_to_pcm16(data)
    else:
        pcm, rate = data, source_rate
    if rate != TWILIO_RATE:
        pcm = resample_pcm16(pcm, rate, TWILIO_RATE)
    return pcm16_to_ulaw(pcm)


def ulaw_duration_ms(ulaw: bytes, sample_rate: int = TWILIO_RATE) -> int:
    return int(len(ulaw) * 1000 / sample_rate)


def split_ulaw(ulaw: bytes, frame_ms: int = FRAME_MS) -> list[bytes]:
    frame = TWILIO_RATE * frame_ms // 1000
    return [ulaw[i : i + frame] for i in range(0, len(ulaw), frame)]


def ulaw_rms(frame: bytes) -> float:
    """RMS energy of a mu-law frame in PCM16 units (silence decodes to ~0)."""
    if not frame:
        return 0.0
    pcm = np.frombuffer(ulaw_to_pcm16(frame), dtype="<i2").astype(np.float64)
    return float(np.sqrt(np.mean(pcm * pcm)))


def tone_ulaw(
    duration_ms: int = 250,
    freq: float = 440.0,
    amplitude: int = 6000,
    sample_rate: int = TWILIO_RATE,
) -> bytes:
    """Deterministic tone used by the mock TTS backend and tests."""
    n = max(1, int(sample_rate * duration_ms / 1000))
    t = np.arange(n, dtype=np.float64) / sample_rate
    envelope = np.minimum(1.0, np.minimum(t * 40.0, (n / sample_rate - t) * 40.0))
    wavef = np.sin(2 * math.pi * freq * t) * amplitude * np.clip(envelope, 0.0, 1.0)
    pcm = np.clip(np.round(wavef), -32768, 32767).astype("<i2").tobytes()
    return pcm16_to_ulaw(pcm)


def speech_duration_ms_for_text(text: str, chars_per_second: float = 14.0) -> int:
    return max(250, int(len(text) / chars_per_second * 1000))
