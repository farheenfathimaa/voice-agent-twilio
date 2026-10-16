"""Run scripted test calls against the voice agent and report quality metrics.

Examples:
    python -m scripts.eval --base-url http://localhost:8000
    python -m scripts.eval --offline --json eval-report.json
    python -m scripts.eval --scenario order --verbose
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

CALLER_POOL = ["+15550001111", "+15550002222", "+15550003333"]


@dataclass
class TurnSpec:
    text: str
    expect: str = "contained"  # contained | handoff
    expect_tool: str | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    sid: str
    category: str
    turns: list[TurnSpec]


SCENARIOS: list[Scenario] = [
    Scenario("ship-time", "faq", [
        TurnSpec("How long does standard shipping take?", keywords=["business day"]),
    ]),
    Scenario("track-order", "tools", [
        TurnSpec("Where is order ORD-1001? I want to track it.",
                 expect_tool="lookup_order", keywords=["shipped"]),
    ]),
    Scenario("return-policy", "faq", [
        TurnSpec("What is your return policy?", keywords=["30 day"]),
    ]),
    Scenario("refund-timing", "faq", [
        TurnSpec("How long do refunds take to show up?", keywords=["business day"]),
    ]),
    Scenario("cancel-order", "faq", [
        TurnSpec("Can I cancel my order before it ships?", keywords=["cancel"]),
    ]),
    Scenario("payment-methods", "faq", [
        TurnSpec("What payment methods do you accept?", keywords=["card"]),
    ]),
    Scenario("duplicate-charge", "faq", [
        TurnSpec("I think I was charged twice for the same order.",
                 keywords=["business day"]),
    ]),
    Scenario("password-reset", "account", [
        TurnSpec("I forgot my password, how do I reset it?", keywords=["password"]),
    ]),
    Scenario("change-email", "account", [
        TurnSpec("How do I change the email address on my account?",
                 keywords=["profile", "account"]),
    ]),
    Scenario("loyalty-points", "account", [
        TurnSpec("How does the loyalty program work?", keywords=["point"]),
    ]),
    Scenario("warranty-length", "warranty", [
        TurnSpec("How long is the warranty on your products?",
                 keywords=["warranty", "year"]),
    ]),
    Scenario("warranty-claim", "warranty", [
        TurnSpec("How do I make a warranty claim?",
                 keywords=["claim", "replacement"]),
    ]),
    Scenario("support-hours", "store", [
        TurnSpec("What are your phone support hours?", keywords=["monday", "friday", "et"]),
    ]),
    Scenario("student-discount", "store", [
        TurnSpec("Do you offer a student discount?", keywords=["percent"]),
    ]),
    Scenario("price-match", "store", [
        TurnSpec("Do you price match other retailers?", keywords=["price", "match"]),
    ]),
    Scenario("international", "faq", [
        TurnSpec("Do you ship internationally?", keywords=["canada"]),
    ]),
    Scenario("unknown-order", "tools", [
        TurnSpec("What is happening with order ORD-9999?",
                 expect="handoff", expect_tool="lookup_order"),
    ]),
    Scenario("human-request", "handoff", [
        TurnSpec("Please connect me to a human agent.", expect="handoff"),
    ]),
    Scenario("bot-check", "handoff", [
        TurnSpec("Are you a real person or a bot?", expect="handoff"),
    ]),
    Scenario("out-of-domain-weather", "handoff", [
        TurnSpec("What is the weather going to be in Tokyo this weekend?", expect="handoff"),
    ]),
    Scenario("out-of-domain-joke", "handoff", [
        TurnSpec("Tell me a joke about penguins.", expect="handoff"),
    ]),
    Scenario("damaged-item-ticket", "tools", [
        TurnSpec("My item arrived damaged, can you open a ticket for me?",
                 expect_tool="create_ticket", keywords=["ticket"]),
    ]),
    Scenario("order-conversation", "multi-turn", [
        TurnSpec("Hi, I have a question about my order."),
        TurnSpec("It is order ORD-2001, where is it?",
                 expect_tool="lookup_order", keywords=["delivered"]),
        TurnSpec("Great, thank you so much!", keywords=["welcome"]),
    ]),
    Scenario("shipping-then-return", "multi-turn", [
        TurnSpec("How long does expedited shipping take?", keywords=["business day", "2"]),
        TurnSpec("If I need to return something, what is your return policy?",
                 keywords=["30 day", "return"]),
        TurnSpec("That is all I needed, thanks!", keywords=["welcome"]),
    ]),
    Scenario("hours-then-human", "multi-turn", [
        TurnSpec("When are you open on Saturdays?", keywords=["saturday", "9", "5"]),
        TurnSpec("Thanks. Now I need to speak to an actual human please.",
                 expect="handoff"),
    ]),
]

TurnResult = dict[str, Any]
CallResult = dict[str, Any]


@dataclass
class EvalReport:
    calls: list[CallResult] = field(default_factory=list)
    turns: list[TurnResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    # ---------------------------------------------------------------- metrics
    @property
    def containment_rate(self) -> float:
        if not self.calls:
            return 0.0
        return sum(1 for c in self.calls if c["contained"]) / len(self.calls)

    @property
    def expectation_match_rate(self) -> float:
        if not self.calls:
            return 0.0
        return sum(1 for c in self.calls if c["matched_expectation"]) / len(self.calls)

    @property
    def avg_latency_ms(self) -> float:
        values = [t["server_total_ms"] for t in self.turns]
        return statistics.fmean(values) if values else 0.0

    @property
    def p95_latency_ms(self) -> float:
        values = sorted(t["server_total_ms"] for t in self.turns)
        if not values:
            return 0.0
        idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
        return values[idx]

    @property
    def avg_client_latency_ms(self) -> float:
        values = [t["client_ms"] for t in self.turns]
        return statistics.fmean(values) if values else 0.0

    @property
    def tool_attempts(self) -> int:
        return sum(t["tool_attempts"] for t in self.turns)

    @property
    def tool_successes(self) -> int:
        return sum(t["tool_successes"] for t in self.turns)

    @property
    def tool_success_rate(self) -> float:
        attempts = self.tool_attempts
        return (self.tool_successes / attempts) if attempts else 1.0

    @property
    def keyword_turns(self) -> int:
        return sum(1 for t in self.turns if t["keywords"])

    @property
    def keyword_hits(self) -> int:
        return sum(1 for t in self.turns if t["keywords"] and t["keywords_ok"])

    @property
    def keyword_accuracy(self) -> float:
        return (self.keyword_hits / self.keyword_turns) if self.keyword_turns else 1.0

    def stage_averages(self) -> dict[str, float]:
        keys = ["stt_ms", "retrieve_ms", "llm_ms", "tool_ms", "tts_ms"]
        out: dict[str, float] = {}
        for key in keys:
            values = [t["server_stages"].get(key, 0.0) for t in self.turns]
            out[key] = round(statistics.fmean(values), 1) if values else 0.0
        return out

    def per_category(self) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[CallResult]] = {}
        for call in self.calls:
            grouped.setdefault(call["category"], []).append(call)
        out: dict[str, dict[str, float]] = {}
        for category, calls in sorted(grouped.items()):
            turn_latencies = [
                t["server_total_ms"] for t in self.turns if t["category"] == category
            ]
            out[category] = {
                "calls": len(calls),
                "containment": round(sum(1 for c in calls if c["contained"]) / len(calls), 3),
                "expectation_match": round(
                    sum(1 for c in calls if c["matched_expectation"]) / len(calls), 3
                ),
                "avg_latency_ms": round(statistics.fmean(turn_latencies), 1)
                if turn_latencies
                else 0.0,
            }
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "turns": len(self.turns),
            "containment_rate": round(self.containment_rate, 3),
            "expectation_match_rate": round(self.expectation_match_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "avg_client_latency_ms": round(self.avg_client_latency_ms, 1),
            "stage_averages_ms": self.stage_averages(),
            "tool_calls_attempted": self.tool_attempts,
            "tool_calls_succeeded": self.tool_successes,
            "tool_success_rate": round(self.tool_success_rate, 3),
            "keyword_accuracy": round(self.keyword_accuracy, 3),
            "per_category": self.per_category(),
            "elapsed_s": round(self.elapsed_s, 2),
            "calls_detail": self.calls,
            "turns_detail": self.turns,
        }


def _run_call(
    client: httpx.Client,
    scenario: Scenario,
    caller_id: str,
    synthesize: bool,
    verbose: bool,
    timeout: float,
) -> tuple[CallResult, list[TurnResult]]:
    session_resp = client.post(
        "/agent/sessions", json={"caller_id": caller_id, "called": "+15551234567"}
    )
    session = session_resp.json()
    call: CallResult = {
        "scenario": scenario.sid,
        "category": scenario.category,
        "caller_id": caller_id,
        "session_id": session["session_id"],
        "call_sid": session["call_sid"],
        "contained": True,
        "handoff": False,
        "matched_expectation": False,
        "turns": 0,
        "keyword_ok": True,
        "tools_attempted": 0,
        "tools_succeeded": 0,
    }
    turn_results: list[TurnResult] = []
    handoff_seen = False

    for index, spec in enumerate(scenario.turns, start=1):
        payload = {
            "text": spec.text,
            "session_id": call["session_id"],
            "caller_id": caller_id,
            "synthesize": synthesize,
            "final": index == len(scenario.turns),
        }
        started = time.perf_counter()
        try:
            resp = client.post("/agent/turn", json=payload, timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            turn_results.append(
                {
                    "scenario": scenario.sid,
                    "category": scenario.category,
                    "turn": index,
                    "text": spec.text,
                    "reply": "",
                    "error": str(exc),
                    "server_total_ms": 0.0,
                    "server_stages": {},
                    "client_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "confidence": 0.0,
                    "handoff": True,
                    "tool_attempts": 0,
                    "tool_successes": 0,
                    "keywords": spec.keywords,
                    "keywords_ok": False,
                    "expect": spec.expect,
                    "expectation_met": False,
                }
            )
            handoff_seen = True
            continue

        data = resp.json()
        client_ms = (time.perf_counter() - started) * 1000.0
        reply = (data.get("reply") or "").lower()
        keywords_ok = all(k.lower() in reply for k in spec.keywords) if spec.keywords else True
        tools = data.get("tools") or []
        tool_ok = [t for t in tools if t.get("ok")]
        tool_names = [t.get("name") for t in tools]
        handoff = bool(data.get("handoff"))
        handoff_seen = handoff_seen or handoff

        if spec.expect_tool:
            expectation_met = spec.expect_tool in tool_names
        elif spec.keywords:
            expectation_met = keywords_ok
        else:
            expectation_met = True

        turn_results.append(
            {
                "scenario": scenario.sid,
                "category": scenario.category,
                "turn": index,
                "text": spec.text,
                "reply": data.get("reply", ""),
                "server_total_ms": float(data["latency"]["total_ms"]),
                "server_stages": data["latency"],
                "client_ms": round(client_ms, 1),
                "confidence": float(data.get("confidence", 0.0)),
                "handoff": handoff,
                "tool_attempts": len(tools),
                "tool_successes": len(tool_ok),
                "expected_tool": spec.expect_tool,
                "tool_names": tool_names,
                "keywords": spec.keywords,
                "keywords_ok": keywords_ok,
                "expect": spec.expect,
                "expectation_met": expectation_met,
                "rag_hits": len(data.get("rag_hits") or []),
            }
        )
        call["turns"] += 1
        call["tools_attempted"] += len(tools)
        call["tools_succeeded"] += len(tool_ok)
        if not keywords_ok:
            call["keyword_ok"] = False
        if verbose:
            print(
                f"    [{scenario.sid} t{index}] {data.get('confidence', 0):.2f} "
                f"{data['latency']['total_ms']:>7.1f}ms  {data.get('reply', '')[:90]}"
            )

    call["contained"] = not handoff_seen
    call["handoff"] = handoff_seen
    expected_handoff = any(t.expect == "handoff" for t in scenario.turns)
    call["matched_expectation"] = call["contained"] != expected_handoff
    call["keyword_ok"] = all(t["keywords_ok"] for t in turn_results if t["keywords"])
    return call, turn_results


def run_eval(
    client: httpx.Client,
    scenarios: list[Scenario] | None = None,
    *,
    synthesize: bool = True,
    verbose: bool = False,
    timeout: float = 60.0,
) -> EvalReport:
    scenarios = scenarios if scenarios is not None else SCENARIOS
    report = EvalReport()
    started = time.perf_counter()
    for offset, scenario in enumerate(scenarios):
        caller_id = CALLER_POOL[offset % len(CALLER_POOL)]
        call, turns = _run_call(client, scenario, caller_id, synthesize, verbose, timeout)
        report.calls.append(call)
        report.turns.extend(turns)
    report.elapsed_s = time.perf_counter() - started
    return report


def print_report(report: EvalReport) -> None:
    stages = report.stage_averages()
    print("")
    print("=" * 68)
    print(f" EVAL REPORT — {len(report.calls)} calls, {len(report.turns)} turns")
    print("=" * 68)
    print(f"  containment rate        : {report.containment_rate:6.1%}")
    print(f"  expectation match rate  : {report.expectation_match_rate:6.1%}")
    print(f"  avg latency / turn      : {report.avg_latency_ms:6.1f} ms")
    print(f"  p95 latency / turn      : {report.p95_latency_ms:6.1f} ms")
    print(f"  avg client-side turn    : {report.avg_client_latency_ms:6.1f} ms")
    print(
        f"  stage avg               : stt {stages['stt_ms']}ms | "
        f"retrieve {stages['retrieve_ms']}ms | llm {stages['llm_ms']}ms | "
        f"tool {stages['tool_ms']}ms | tts {stages['tts_ms']}ms"
    )
    print(
        f"  tool-call success       : {report.tool_successes}/"
        f"{report.tool_attempts} ({report.tool_success_rate:.1%})"
    )
    print(
        f"  answer keyword accuracy : {report.keyword_hits}/"
        f"{report.keyword_turns} ({report.keyword_accuracy:.1%})"
    )
    print(f"  wall time               : {report.elapsed_s:.1f}s")
    print("-" * 68)
    print(f"  {'category':<16} {'calls':>5} {'contain':>8} {'match':>7} {'lat ms':>8}")
    for category, stats in report.per_category().items():
        print(
            f"  {category:<16} {stats['calls']:>5} {stats['containment']:>8.1%} "
            f"{stats['expectation_match']:>7.1%} {stats['avg_latency_ms']:>8.1f}"
        )
    failures = [c for c in report.calls if not c["matched_expectation"] or not c["keyword_ok"]]
    if failures:
        print("-" * 68)
        print("  calls needing attention:")
        for call in failures:
            reasons = []
            if not call["matched_expectation"]:
                reasons.append("handoff mismatch")
            if not call["keyword_ok"]:
                reasons.append("missing keywords")
            print(f"    - {call['scenario']:<24} {', '.join(reasons)}")
    print("=" * 68)


def offline_settings(**overrides: Any):
    from app.config import Settings

    base: dict[str, Any] = {
        "_env_file": None,
        "environment": "eval-offline",
        "llm_provider": "mock",
        "stt_provider": "mock",
        "tts_provider": "mock",
        "embeddings_provider": "hashing",
        "n8n_webhook_url": "",
        "public_base_url": "http://eval.local",
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(**base)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in scenarios if args.scenario in s.sid]
    if not scenarios:
        print(f"no scenarios match {args.scenario!r}", file=sys.stderr)
        return 2

    synthesize = not args.no_tts
    if args.offline:
        from fastapi.testclient import TestClient

        from app.main import create_app

        app = create_app(offline_settings())
        with TestClient(app) as client:
            try:
                from mock_crm.server import app as crm_app

                app.state.c.crm.use_asgi(crm_app)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: mock CRM not mounted: {exc}", file=sys.stderr)
            report = run_eval(
                client,
                scenarios,
                synthesize=synthesize,
                verbose=args.verbose,
                timeout=args.timeout,
            )
    else:
        try:
            with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
                report = run_eval(
                    client,
                    scenarios,
                    synthesize=synthesize,
                    verbose=args.verbose,
                    timeout=args.timeout,
                )
        except httpx.HTTPError as exc:
            print(f"cannot reach {args.base_url}: {exc}", file=sys.stderr)
            print("is the agent running? try: docker compose up --build", file=sys.stderr)
            return 2

    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if report.expectation_match_rate >= args.min_match else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scripted eval for the voice agent")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--offline", action="store_true",
                        help="run in-process with mock providers (no API keys needed)")
    parser.add_argument("--scenario", default="", help="only run scenarios whose id contains this")
    parser.add_argument("--json", default="", help="write the full report to this path")
    parser.add_argument("--no-tts", action="store_true", help="skip text-to-speech synthesis")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--min-match", type=float, default=0.9,
                        help="exit non-zero below this expectation match rate")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
