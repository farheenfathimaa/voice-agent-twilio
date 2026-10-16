from __future__ import annotations

import pytest

from app.confidence import detect_handoff, is_smalltalk, score_confidence
from app.rag import Chunk, Hit
from app.sessions import ToolRecord


def _hit(score: float) -> Hit:
    chunk = Chunk(
        doc_id="shipping.md#0",
        title="Shipping",
        text="x",
        source="shipping.md",
        order=0,
    )
    return Hit(chunk=chunk, score=score)


@pytest.mark.parametrize(
    "text",
    [
        "Please connect me to a human agent.",
        "I want to speak to a real person.",
        "Can I speak with a specialist?",
        "I need to talk to someone.",
        "Escalate this please.",
        "Are you a real person or a bot?",
        "Press zero to reach an operator.",
    ],
)
def test_detect_handoff_matches(text: str) -> None:
    assert detect_handoff(text)


@pytest.mark.parametrize(
    "text",
    [
        "How long does standard shipping take?",
        "Where is my order ORD-1001?",
        "What is your return policy?",
    ],
)
def test_detect_handoff_ignores_normal_requests(text: str) -> None:
    assert detect_handoff(text) is None


@pytest.mark.parametrize(
    "text",
    ["hi", "hello there", "thanks", "thank you so much", "okay", "that is all I needed, thanks!",
     "great", "no thanks, I'm good", "bye"],
)
def test_is_smalltalk_positive(text: str) -> None:
    assert is_smalltalk(text)


@pytest.mark.parametrize(
    "text",
    ["How long is standard shipping?", "Where is my order?", "Do you offer a student discount?"],
)
def test_is_smalltalk_negative(text: str) -> None:
    assert not is_smalltalk(text)


def test_handoff_intent_scores_zero() -> None:
    assert score_confidence(
        utterance="connect me to a human",
        reply="Of course.",
        hits=[],
        provider="hashing",
        min_score=0.34,
        handoff_intent="explicit request",
    ) == 0.0


def test_tool_success_is_high_confidence() -> None:
    records = [ToolRecord(name="lookup_order", arguments={}, ok=True, latency_ms=10)]
    score = score_confidence(
        "where is my order", "it shipped", [], "hashing", 0.34, tool_records=records
    )
    assert score == pytest.approx(0.92)


def test_tool_failure_is_low_confidence() -> None:
    records = [ToolRecord(name="lookup_order", arguments={}, ok=False, latency_ms=10, error="404")]
    score = score_confidence(
        "where is order ORD-9999",
        "I could not pull that up",
        [],
        "hashing",
        0.34,
        tool_records=records,
    )
    assert score == pytest.approx(0.3)


def test_smalltalk_floor() -> None:
    score = score_confidence("thanks!", "You are welcome", [], "hashing", 0.34)
    assert score == pytest.approx(0.85)


def test_ungrounded_question_is_low() -> None:
    score = score_confidence("weather in tokyo", "no idea", [], "hashing", 0.34)
    assert score == pytest.approx(0.3)


def test_grounded_hit_scales_confidence() -> None:
    strong = score_confidence("shipping time", "answer", [_hit(0.72)], "hashing", 0.34)
    weak = score_confidence("shipping time", "answer", [_hit(0.4)], "hashing", 0.34)
    assert strong > weak > 0.35
    assert strong == pytest.approx(1.0)


def test_refusal_caps_confidence() -> None:
    reply = "I do not have that information."
    score = score_confidence("shipping?", reply, [_hit(0.72)], "hashing", 0.34)
    assert score == pytest.approx(0.35)


def test_low_stt_confidence_caps_score() -> None:
    score = score_confidence(
        "shipping time", "answer", [_hit(0.72)], "hashing", 0.34, stt_confidence=0.2
    )
    assert score == pytest.approx(0.4)
