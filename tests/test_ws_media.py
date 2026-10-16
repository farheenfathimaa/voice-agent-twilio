from __future__ import annotations

import base64
import json
import time

from fastapi.testclient import TestClient

from app import audio
from app.streams import MediaConnection

LOUD = audio.tone_ulaw(20, amplitude=8000)
SILENCE = audio.pcm16_to_ulaw(bytes(320))


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _media(payload: bytes) -> dict:
    return {"event": "media", "media": {"payload": _b64(payload)}}


def _start(stream_sid: str, call_sid: str) -> dict:
    return {
        "event": "start",
        "streamSid": stream_sid,
        "start": {
            "callSid": call_sid,
            "streamSid": stream_sid,
            "customParameters": {
                "caller_id": "+15550001111",
                "call_sid": call_sid,
                "called": "+15551230000",
            },
        },
    }


def _drain_until_mark(ws, stream_sid: str, limit: int = 4000) -> dict:
    seen = {"media": 0, "clear": 0}
    for _ in range(limit):
        message = ws.receive_json()
        event = message.get("event")
        if event == "media":
            seen["media"] += 1
            assert message.get("streamSid") == stream_sid
        elif event == "clear":
            seen["clear"] += 1
        elif event == "mark":
            ws.send_json({"event": "mark", "mark": message["mark"]})
            return seen
    raise AssertionError("no mark received within message budget")


def test_media_stream_completes_a_turn(client: TestClient, components) -> None:
    components.stt.push("What is your return policy?")

    with client.websocket_connect("/media") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json(_start("MZ-stream-1", "CAstream01"))

        for _ in range(12):
            ws.send_json(_media(LOUD))
        time.sleep(0.8)
        ws.send_json(_media(SILENCE))

        seen = _drain_until_mark(ws, "MZ-stream-1")
        assert seen["media"] > 0
        time.sleep(0.5)  # let the server commit the finished turn

    session = components.sessions.get("CAstream01")
    assert session is not None
    assert session.stream_sid == "MZ-stream-1"
    assert len(session.turns) == 1
    assert session.turns[0].user_text == "What is your return policy?"
    assert session.turns[0].reply
    assert components.stt.calls, "STT was never invoked"


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.close_code: int | None = None

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000) -> None:
        self.close_code = code


async def test_dtmf_zero_triggers_transfer(components, settings) -> None:
    ws = FakeWebSocket()
    conn = MediaConnection(ws, settings, components)
    conn.stream_sid = "MZ-dtmf"
    conn.session = components.sessions.create(
        caller_id="+15550001111", call_sid="CAdtmf01", source="stream"
    )

    async def fake_play(ulaw: bytes, tag: str) -> bool:
        return True

    conn._play = fake_play  # type: ignore[assignment]

    await conn._on_dtmf({"dtmf": {"digit": "0"}})
    assert conn.turn_task is not None
    await conn.turn_task

    assert conn.session.transfer_requested is True
    assert conn.session.transfer_reason == "dtmf 0"
    assert ws.close_code == 1000
    assert any(msg["event"] == "clear" for msg in ws.sent)


async def test_non_handoff_dtmf_is_ignored(components, settings) -> None:
    ws = FakeWebSocket()
    conn = MediaConnection(ws, settings, components)
    conn.stream_sid = "MZ-dtmf2"
    conn.session = components.sessions.create(
        caller_id="+15550001111", call_sid="CAdtmf02", source="stream"
    )

    await conn._on_dtmf({"dtmf": {"digit": "5"}})
    assert conn.turn_task is None
    assert conn.session.transfer_requested is False
    assert ws.sent == []
