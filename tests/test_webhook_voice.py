from __future__ import annotations

from fastapi.testclient import TestClient


def _post_voice(client: TestClient, call_sid: str = "CAtestcall01") -> str:
    resp = client.post(
        "/voice",
        data={"CallSid": call_sid, "From": "+15550001111", "To": "+15551230000"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    return resp.text


def test_voice_webhook_returns_stream_connect(client: TestClient, components) -> None:
    xml = _post_voice(client)
    assert "<Connect" in xml
    assert 'action="http://test.local/voice/connect"' in xml
    assert 'url="ws://test.local/media"' in xml
    assert 'name="caller_id" value="+15550001111"' in xml
    assert 'name="call_sid" value="CAtestcall01"' in xml
    assert 'name="called" value="+15551230000"' in xml

    session = components.sessions.get("CAtestcall01")
    assert session is not None
    assert session.caller_id == "+15550001111"
    assert session.called == "+15551230000"
    assert session.source == "stream"


def test_connect_action_without_transfer_has_no_dial(client: TestClient) -> None:
    _post_voice(client, "CAtestcall02")
    resp = client.post("/voice/connect", data={"CallSid": "CAtestcall02"})
    assert resp.status_code == 200
    assert "<Dial" not in resp.text


def test_connect_action_with_transfer_dials_number(client: TestClient, components) -> None:
    _post_voice(client, "CAtestcall03")
    session = components.sessions.get("CAtestcall03")
    session.transfer_requested = True
    session.transfer_reason = "explicit request"

    resp = client.post("/voice/connect", data={"CallSid": "CAtestcall03"})
    assert resp.status_code == 200
    assert "<Dial" in resp.text
    assert "<Number>+15559876543</Number>" in resp.text
    assert 'callerId="+15551230000"' in resp.text


def test_connect_action_unknown_call(client: TestClient) -> None:
    resp = client.post("/voice/connect", data={"CallSid": "CAdoesnotexist"})
    assert resp.status_code == 200
    assert "<Dial" not in resp.text
