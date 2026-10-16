from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("voice.sessions")

EndHandler = Callable[["CallSession"], Awaitable[None]]


def new_call_sid(prefix: str = "CAtest") -> str:
    return f"{prefix}{uuid.uuid4().hex[:20]}"


@dataclass
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    latency_ms: float
    error: str | None = None
    result: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


@dataclass
class TurnRecord:
    index: int
    user_text: str
    reply: str
    confidence: float
    handoff: bool
    latency_ms: float
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "user": self.user_text,
            "assistant": self.reply,
            "confidence": round(self.confidence, 3),
            "handoff": self.handoff,
            "latency_ms": round(self.latency_ms, 1),
            "ts": self.ts,
        }


@dataclass
class CallSession:
    call_sid: str
    session_id: str
    caller_id: str = ""
    called: str = ""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    stream_sid: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    tool_records: list[ToolRecord] = field(default_factory=list)
    transfer_requested: bool = False
    transfer_reason: str = ""
    summary_posted: bool = False
    active: bool = False
    ended: bool = False
    source: str = "http"

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def duration_s(self) -> float:
        return max(0.0, time.time() - self.created_at)

    @property
    def contained(self) -> bool:
        return not self.transfer_requested

    @property
    def avg_latency_ms(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.latency_ms for t in self.turns) / len(self.turns)

    def touch(self) -> None:
        self.last_activity = time.time()

    def metadata(self) -> dict[str, Any]:
        return {
            "caller_id": self.caller_id,
            "call_sid": self.call_sid,
            "session_id": self.session_id,
            "called": self.called,
        }

    def as_dict(self, include_transcript: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            **self.metadata(),
            "started_at": self.created_at,
            "duration_s": round(self.duration_s, 2),
            "turns": len(self.turns),
            "contained": self.contained,
            "handoff": self.transfer_requested,
            "transfer_reason": self.transfer_reason,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "tool_calls": [t.as_dict() for t in self.tool_records],
            "source": self.source,
        }
        if include_transcript:
            data["transcript"] = [t.as_dict() for t in self.turns]
        return data


class SessionRegistry:
    def __init__(self, ttl_s: int = 1800) -> None:
        self._sessions: dict[str, CallSession] = {}
        self._index: dict[str, str] = {}
        self._ttl_s = ttl_s
        self._end_handler: EndHandler | None = None

    def set_end_handler(self, handler: EndHandler) -> None:
        self._end_handler = handler

    def create(
        self,
        caller_id: str = "",
        call_sid: str | None = None,
        called: str = "",
        source: str = "http",
    ) -> CallSession:
        sid = call_sid or new_call_sid()
        session = CallSession(
            call_sid=sid,
            session_id=sid,
            caller_id=caller_id,
            called=called,
            source=source,
        )
        self._sessions[sid] = session
        self._index[session.session_id] = sid
        return session

    def get(self, key: str | None) -> CallSession | None:
        if not key:
            return None
        if key in self._sessions:
            return self._sessions[key]
        sid = self._index.get(key)
        return self._sessions.get(sid) if sid else None

    def get_or_create(
        self,
        call_sid: str | None = None,
        session_id: str | None = None,
        caller_id: str = "",
        called: str = "",
        source: str = "http",
    ) -> CallSession:
        existing = self.get(session_id) or self.get(call_sid)
        if existing is not None:
            if caller_id and not existing.caller_id:
                existing.caller_id = caller_id
            existing.touch()
            return existing
        session = self.create(caller_id=caller_id, call_sid=call_sid, called=called, source=source)
        session.session_id = session_id or session.session_id
        self._index[session.session_id] = session.call_sid
        return session

    def all(self) -> list[CallSession]:
        return list(self._sessions.values())

    async def finalize(self, session: CallSession, reason: str = "ended") -> None:
        if session.ended:
            return
        session.ended = True
        session.active = False
        session.touch()
        if self._end_handler is None:
            return
        try:
            await self._end_handler(session)
            session.summary_posted = True
        except Exception:  # noqa: BLE001 - summary failures must never kill a call
            logger.exception("end handler failed for %s (%s)", session.call_sid, reason)

    async def cleanup_expired(self, now: float | None = None) -> int:
        now = now or time.time()
        expired = [
            s
            for s in self._sessions.values()
            if not s.ended and not s.active and now - s.last_activity > self._ttl_s
        ]
        for session in expired:
            await self.finalize(session, reason="ttl")
            self._sessions.pop(session.call_sid, None)
            self._index.pop(session.session_id, None)
        return len(expired)
