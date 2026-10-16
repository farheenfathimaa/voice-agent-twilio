from __future__ import annotations

import json
import logging
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("voice.latency")


@dataclass
class TurnMetrics:
    turn_index: int
    call_sid: str = ""
    session_id: str = ""
    caller_id: str = ""
    vad_ms: float = 0.0
    stt_ms: float = 0.0
    retrieve_ms: float = 0.0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    confidence: float = 0.0
    handoff: bool = False
    interrupted: bool = False
    tool_calls: int = 0
    reply_chars: int = 0
    user_chars: int = 0
    model: str = ""
    stt_provider: str = ""
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("vad_ms", "stt_ms", "retrieve_ms", "llm_ms", "tool_ms", "tts_ms", "total_ms"):
            data[key] = round(float(data[key]), 1)
        data["confidence"] = round(float(data["confidence"]), 3)
        return data

    @property
    def stage_sum_ms(self) -> float:
        return self.stt_ms + self.retrieve_ms + self.llm_ms + self.tool_ms + self.tts_ms


class MetricsStore:
    def __init__(self, window: int = 500) -> None:
        self._turns: deque[TurnMetrics] = deque(maxlen=window)

    def record(self, metrics: TurnMetrics) -> None:
        self._turns.append(metrics)
        logger.info(json.dumps({"event": "turn_latency", **metrics.as_dict()}))

    def recent(self, limit: int = 100) -> list[TurnMetrics]:
        turns = list(self._turns)
        return turns[-limit:]

    def summary(self) -> dict[str, Any]:
        turns = list(self._turns)
        if not turns:
            return {"turns": 0}
        totals = [t.total_ms for t in turns]
        return {
            "turns": len(turns),
            "avg_total_ms": round(statistics.fmean(totals), 1),
            "p50_total_ms": round(statistics.median(totals), 1),
            "p95_total_ms": round(_percentile(totals, 95), 1),
            "avg_stt_ms": round(statistics.fmean(t.stt_ms for t in turns), 1),
            "avg_retrieve_ms": round(statistics.fmean(t.retrieve_ms for t in turns), 1),
            "avg_llm_ms": round(statistics.fmean(t.llm_ms for t in turns), 1),
            "avg_tool_ms": round(statistics.fmean(t.tool_ms for t in turns), 1),
            "avg_tts_ms": round(statistics.fmean(t.tts_ms for t in turns), 1),
            "handoffs": sum(1 for t in turns if t.handoff),
            "avg_confidence": round(statistics.fmean(t.confidence for t in turns), 3),
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    frac = k - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


class StageTimer:
    """Context manager measuring a pipeline stage in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
