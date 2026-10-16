from __future__ import annotations

import numpy as np
import pytest

from app import audio


def test_ulaw_roundtrip_is_close() -> None:
    samples = (np.sin(np.linspace(0, 40, 400)) * 12000).astype("<i2")
    pcm = samples.tobytes()
    ulaw = audio.pcm16_to_ulaw(pcm)
    assert len(ulaw) == samples.size
    decoded = np.frombuffer(audio.ulaw_to_pcm16(ulaw), dtype="<i2")
    assert decoded.size == samples.size
    # mu-law is lossy but should track the waveform within ~2% full scale
    assert np.max(np.abs(decoded.astype(np.int32) - samples.astype(np.int32))) < 1400


def test_silence_decodes_to_zero_rms() -> None:
    silence = audio.pcm16_to_ulaw(bytes(320))
    assert audio.ulaw_rms(silence) == pytest.approx(0.0, abs=1.0)


def test_tone_has_energy() -> None:
    tone = audio.tone_ulaw(100)
    assert audio.ulaw_rms(tone) > 1000


def test_duration_and_split() -> None:
    tone = audio.tone_ulaw(100)
    assert audio.ulaw_duration_ms(tone) == pytest.approx(100, abs=1)
    frames = audio.split_ulaw(tone, frame_ms=20)
    assert len(frames) == 5
    assert all(len(f) == 160 for f in frames)


def test_split_ulaw_trailing_partial_frame() -> None:
    payload = bytes(170)
    frames = audio.split_ulaw(payload, frame_ms=20)
    assert [len(f) for f in frames] == [160, 10]


def test_resample_identity_and_decimation() -> None:
    pcm = audio.tone_ulaw(20)
    assert audio.resample_pcm16(pcm, 8000, 8000) == pcm

    source = (np.arange(320, dtype="<i2")).tobytes()
    down = audio.resample_pcm16(source, 16000, 8000)
    assert len(down) == len(source) // 2


def test_wav_roundtrip() -> None:
    pcm = audio.tone_ulaw(20)
    raw = audio.pcm16_to_ulaw(pcm)
    decoded = audio.ulaw_to_pcm16(raw)
    wav = audio.build_wav(decoded, audio.TWILIO_RATE)
    back, rate = audio.wav_to_pcm16(wav)
    assert rate == audio.TWILIO_RATE
    assert back == decoded


def test_audio_to_ulaw8k_from_wav() -> None:
    pcm = audio.ulaw_to_pcm16(audio.tone_ulaw(40))
    wav = audio.build_wav(pcm, 16000)
    ulaw = audio.audio_to_ulaw8k(wav)
    assert ulaw
    assert audio.ulaw_duration_ms(ulaw) == pytest.approx(20, abs=2)


def test_audio_to_ulaw8k_rejects_non_wav() -> None:
    with pytest.raises(ValueError):
        audio.audio_to_ulaw8k(b"not a wav")


def test_speech_duration_scales_with_text() -> None:
    assert audio.speech_duration_ms_for_text("") == 250
    assert audio.speech_duration_ms_for_text("x" * 140) > audio.speech_duration_ms_for_text("x")
