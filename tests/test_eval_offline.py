from __future__ import annotations

import pytest

from scripts.eval import SCENARIOS, run_eval


def _subset(*sids: str):
    return [s for s in SCENARIOS if s.sid in sids]


def test_offline_eval_containment_scenarios(client) -> None:
    scenarios = _subset(
        "ship-time",
        "track-order",
        "return-policy",
        "student-discount",
        "human-request",
        "out-of-domain-weather",
    )
    report = run_eval(client, scenarios, synthesize=False, timeout=30.0)

    assert len(report.calls) == len(scenarios)
    assert report.expectation_match_rate == 1.0
    assert report.keyword_accuracy == 1.0
    # 2 of the 6 scenarios are expected to be transferred to a human
    assert report.containment_rate == pytest.approx(4 / 6)
    assert report.tool_success_rate == 1.0


def test_offline_eval_reports_latency(client) -> None:
    report = run_eval(client, _subset("ship-time"), synthesize=False, timeout=30.0)
    assert report.avg_latency_ms >= 0.0
    assert report.p95_latency_ms >= 0.0
    assert report.as_dict()["calls"] == 1
