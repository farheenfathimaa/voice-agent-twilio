from __future__ import annotations

import base64

from fastapi.testclient import TestClient


def test_session_creation(client: TestClient) -> None:
    resp = client.post("/agent/sessions", json={"caller_id": "+15550001111"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["session_id"] == body["call_sid"]
    assert body["caller_id"] == "+15550001111"


def test_faq_turn_is_contained(turn, new_session) -> None:
    session = new_session()
    body = turn("How long does standard shipping take?", session)
    assert body["handoff"] is False
    assert body["contained"] is True
    assert "business day" in body["reply"].lower()
    assert body["confidence"] >= 0.6
    assert body["latency"]["total_ms"] >= 0.0
    assert body["turn"] == 1


def test_explicit_handoff_uses_ack(turn, new_session, components) -> None:
    session = new_session()
    body = turn("Please connect me to a human agent.", session)
    assert body["handoff"] is True
    assert body["contained"] is False
    assert body["reply"] == components.settings.handoff_ack
    assert body["confidence"] == 0.0


def test_out_of_domain_hands_off(turn, new_session) -> None:
    session = new_session()
    body = turn("What is the weather in Tokyo this weekend?", session)
    assert body["handoff"] is True


def test_order_lookup_uses_tool(turn, new_session) -> None:
    session = new_session()
    body = turn("Where is order ORD-1001? I want to track it.", session)
    assert body["handoff"] is False
    assert body["tools"], "expected a tool call"
    record = body["tools"][0]
    assert record["name"] == "lookup_order"
    assert record["ok"] is True
    assert "shipped" in body["reply"].lower()


def test_unknown_order_hands_off(turn, new_session) -> None:
    session = new_session()
    body = turn("What is happening with order ORD-9999?", session)
    assert body["tools"] and body["tools"][0]["ok"] is False
    assert body["handoff"] is True


def test_create_ticket_tool(turn, new_session) -> None:
    session = new_session()
    body = turn("My item arrived damaged, can you open a ticket for me?", session)
    assert body["tools"] and body["tools"][0]["name"] == "create_ticket"
    assert body["tools"][0]["ok"] is True
    assert "ticket" in body["reply"].lower()


def test_smalltalk_is_contained(turn, new_session) -> None:
    session = new_session()
    body = turn("Great, thank you so much!", session)
    assert body["handoff"] is False
    assert body["confidence"] >= 0.6


def test_audio_is_returned_when_requested(turn, new_session) -> None:
    session = new_session()
    body = turn("Thanks!", session, synthesize=True, include_audio=True)
    assert body["audio_b64"]
    decoded = base64.b64decode(body["audio_b64"])
    assert decoded
    assert body["latency"]["tts_ms"] >= 0.0


def test_final_turn_posts_summary(turn, new_session, notifications) -> None:
    session = new_session(caller_id="+15550001111")
    turn("How long does standard shipping take?", session, final=True)
    assert len(notifications) == 1
    payload = notifications[0]
    assert payload["event"] == "call_summary"
    assert payload["caller_id"] == "+15550001111"
    assert payload["contained"] is True
    assert payload["turn_count"] == 1
    assert payload["topics"] == ["shipping"]


def test_end_call_reports_summary(client: TestClient, turn, new_session) -> None:
    session = new_session()
    turn("What is your return policy?", session, final=True)
    resp = client.post(f"/calls/{session['call_sid']}/end")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary_posted"] is True
    assert body["turns"] == 1
    assert body["handoff"] is False


def test_end_unknown_call_404(client: TestClient) -> None:
    resp = client.post("/calls/CAmissing/end")
    assert resp.status_code == 404


def test_handoff_finalizer_posts_summary(turn, new_session, notifications) -> None:
    session = new_session()
    turn("Please connect me to a human agent.", session, final=True)
    assert len(notifications) == 1
    assert notifications[0]["handoff"] is True
