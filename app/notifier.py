from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from .config import Settings
from .sessions import CallSession

logger = logging.getLogger("voice.n8n")

_TOPIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bord[-\s]?\d+\b", re.I), "order status"),
    (re.compile(r"\b(ship|shipping|deliver|delivery|arrive|dispatch)\b", re.I), "shipping"),
    (re.compile(r"\b(refund|return|money back|cancel my order)\b", re.I), "returns/refund"),
    (re.compile(r"\b(bill|charge|payment|invoice|price|cost|plan)\b", re.I), "billing"),
    (re.compile(r"\b(ticket|complaint|complain|escalat)\b", re.I), "ticket"),
    (re.compile(r"\b(warranty|broken|damaged|defective|repair)\b", re.I), "warranty"),
    (re.compile(r"\b(account|password|login|email|profile)\b", re.I), "account"),
]


def extract_topics(session: CallSession) -> list[str]:
    text = " ".join([t.user_text for t in session.turns]).lower()
    topics = [label for pattern, label in _TOPIC_PATTERNS if pattern.search(text)]
    seen: set[str] = set()
    unique = []
    for topic in topics:
        if topic not in seen:
            seen.add(topic)
            unique.append(topic)
    return unique or ["general inquiry"]


def build_summary_text(session: CallSession) -> str:
    topics = ", ".join(extract_topics(session))
    outcome = "transferred to a human" if session.transfer_requested else "contained by the agent"
    first = session.turns[0].user_text if session.turns else "(no speech transcribed)"
    return (
        f"{session.turn_count}-turn {session.source} call from {session.caller_id or 'unknown'} "
        f"regarding {topics}; {outcome}. Opening issue: {first}"
    )


def build_payload(session: CallSession, settings: Settings, summary: str) -> dict[str, Any]:
    latencies = [t.latency_ms for t in session.turns]
    return {
        "event": "call_summary",
        "generated_at": time.time(),
        "agent": settings.agent_name,
        "business": settings.business_name,
        "llm_model": settings.llm_model,
        **session.metadata(),
        "started_at": session.created_at,
        "duration_s": round(session.duration_s, 2),
        "turn_count": session.turns.__len__(),
        "contained": session.contained,
        "handoff": session.transfer_requested,
        "transfer_reason": session.transfer_reason,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 1) if latencies else 0.0,
        "topics": extract_topics(session),
        "summary": summary,
        "transcript": [
            {"role": "user", "content": t.user_text, "confidence": t.confidence}
            for t in session.turns
        ]
        + [{"role": "assistant", "content": t.reply} for t in session.turns],
        "tool_calls": [t.as_dict() for t in session.tool_records],
    }


class SummaryNotifier:
    """Posts a call summary to an n8n webhook. Failures never propagate."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self.delivered: list[dict[str, Any]] = []

    async def close(self) -> None:
        await self._client.aclose()

    async def notify(self, session: CallSession) -> bool:
        url = self.settings.n8n_webhook_url
        summary = build_summary_text(session)
        payload = build_payload(session, self.settings, summary)
        if not url:
            logger.debug("n8n webhook not configured; summary for %s skipped", session.call_sid)
            return False
        try:
            resp = await self._client.post(url, json=payload, timeout=10.0)
        except httpx.HTTPError as exc:
            logger.error("n8n webhook failed for %s: %s", session.call_sid, exc)
            return False
        if resp.status_code >= 400:
            logger.error(
                "n8n webhook returned %s for %s: %s",
                resp.status_code,
                session.call_sid,
                resp.text[:200],
            )
            return False
        logger.info("n8n summary posted for %s", session.call_sid)
        self.delivered.append(payload)
        return True
