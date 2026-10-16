"""Turn-level confidence scoring and human-handoff detection."""

from __future__ import annotations

import re

from .rag import Hit
from .sessions import ToolRecord

# blended retrieval score ranges per embedding provider: (min relevant, clearly grounded)
PROVIDER_RANGES: dict[str, tuple[float, float]] = {
    "fastembed": (0.42, 0.80),
    "hashing": (0.30, 0.72),
}

HANDOFF_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(speak|talk|chat|connect|put|transfer)\b[^.?!]{0,30}"
        r"\b(human|agent|person|people|someone|somebody|representative|operator|specialist)\b",
        r"\b(human|live|real)\s+(agent|support|help|person|being|assistant)\b",
        r"\bi (?:want|need|would like) (?:to (?:speak|talk|chat) with|a|an|the)\b",
        r"\bget me (?:a|an|the|someone|somebody)\b",
        r"\bescalat(?:e|ion|ing)\b",
        r"\bthis is (?:a )?(?:robot|bot|ai|machine|computer)\b",
        r"\byou are (?:a )?(?:bot|robot|ai)\b",
        r"\b(?:are|is) you (?:a |an |the )?(?:bot|robot|ai|human|person|machine)\b",
        r"\b(?:am i|is this) (?:talking to|speaking with) (?:a |an |the )?"
        r"(?:bot|robot|ai|human|person)\b",
        r"\bpress (?:zero|0|nine|9)\b",
        r"\banyone else (?:there|available)\b",
    )
]

SMALLTALK_PATTERN = re.compile(
    r"^(hi|hello|hey|yo|good (morning|afternoon|evening)|howdy|thanks?|thank you|thx|"
    r"ok(?:ay)?|sure|yes|yeah|no|nope|bye|goodbye|see you|hold on|please hold|"
    r"great|perfect|cool|appreciate it|that's all|nothing else|not right now)[.!?\s]*$",
    re.IGNORECASE,
)

# an utterance made *only* of conversational filler words is smalltalk, e.g.
# "great, thank you so much" or "that is all I needed, thanks!"
# deliberately excludes request words (help, can, want, order, ...) so real
# questions never qualify.
SMALLTALK_VOCAB = {
    # greetings
    "hi", "hey", "hello", "yo", "howdy", "good", "morning", "afternoon", "evening",
    # thanks / politeness
    "thank", "thanks", "thx", "appreciate", "appreciated", "welcome", "please",
    # agreement / acknowledgement
    "ok", "okay", "sure", "yes", "yeah", "no", "nope", "right", "fine", "well",
    # farewells
    "bye", "goodbye", "see", "soon", "later",
    # enthusiasm
    "great", "perfect", "cool", "awesome", "super", "excellent", "nice", "lovely",
    # degree / filler
    "so", "very", "much", "really", "just", "all", "else", "anything", "nothing",
    "something", "now", "again", "done", "finished", "set",
    # function words needed to build short utterances
    "i", "my", "me", "you", "your", "we", "it", "is", "am", "are", "a", "an",
    "the", "that", "this", "there", "to", "of", "for", "with", "and", "or",
    "at", "as", "in", "on", "be", "do", "not", "that's", "thats", "i'm", "im",
    # state words
    "need", "needed", "have", "had",
}

REFUSAL_PATTERN = re.compile(
    r"(i (?:do not|don't) (?:have|know|have any)|i'm not sure|i am not sure|"
    r"i (?:cannot|can't) (?:help|assist|find|answer)|unable to find|"
    r"no (?:information|record) (?:found|for)|"
    r"not in (?:my|our) (?:knowledge|records)|doesn't appear|does not appear)",
    re.IGNORECASE,
)


def detect_handoff(text: str) -> str | None:
    """Return a reason string when the caller explicitly asks for a human."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return None
    for pattern in HANDOFF_PATTERNS:
        if pattern.search(cleaned):
            return f"explicit request: {pattern.pattern[:60]}"
    return None


def is_smalltalk(text: str) -> bool:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return True
    if len(cleaned) > 60:
        return False
    if SMALLTALK_PATTERN.match(cleaned):
        return True
    tokens = re.findall(r"[a-z']+", cleaned.lower())
    return bool(tokens) and all(t.strip("'") in SMALLTALK_VOCAB for t in tokens)


def score_confidence(
    utterance: str,
    reply: str,
    hits: list[Hit],
    provider: str,
    min_score: float,
    tool_records: list[ToolRecord] | None = None,
    handoff_intent: str | None = None,
    stt_confidence: float = 1.0,
) -> float:
    """0..1 estimate of whether the reply is trustworthy enough to contain the call."""
    if handoff_intent:
        return 0.0

    tool_records = tool_records or []
    if any(r.ok for r in tool_records):
        score = 0.92
    elif tool_records:
        score = 0.3
    elif is_smalltalk(utterance):
        score = 0.85
    else:
        grounded = [h for h in hits if h.score >= min_score]
        if not grounded:
            score = 0.3
        else:
            lo, hi = PROVIDER_RANGES.get(provider, (0.35, 0.7))
            sim = grounded[0].score
            base = min(max((sim - lo) / max(hi - lo, 1e-6), 0.0), 1.0)
            score = 0.35 + 0.65 * base

    if reply and REFUSAL_PATTERN.search(reply):
        score = min(score, 0.35)
    if stt_confidence < 0.35:
        score = min(score, 0.4)
    return round(min(max(score, 0.0), 1.0), 3)
