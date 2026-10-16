from __future__ import annotations

from app import twiml


def test_response_wraps_and_escapes() -> None:
    xml = twiml.response(twiml.say("Tom & Jerry"))
    assert xml.startswith("<?xml")
    assert xml.endswith("</Response>")
    assert "Tom &amp; Jerry" in xml


def test_connect_stream_contains_stream_and_action() -> None:
    xml = twiml.response(
        twiml.connect_stream(
            "wss://host.example/media",
            action_url="https://host.example/voice/connect",
            parameters={"caller_id": "+15550001111", "call_sid": "CA123", "called": ""},
        )
    )
    assert "<Connect action=\"https://host.example/voice/connect\"" in xml
    assert "<Stream url=\"wss://host.example/media\"" in xml
    assert "name=\"caller_id\" value=\"+15550001111\"" in xml
    assert "name=\"call_sid\" value=\"CA123\"" in xml
    # a present-but-empty parameter is still emitted (None would drop it)
    assert "name=\"called\" value=\"\"" in xml


def test_transfer_twiml_has_say_and_dial() -> None:
    xml = twiml.transfer_twiml(
        "+15559876543", "Please hold", caller_id="+15551230000", timeout=15
    )
    assert "<Say" in xml and "Please hold" in xml
    assert "<Dial" in xml
    assert "<Number>+15559876543</Number>" in xml
    assert "callerId=\"+15551230000\"" in xml
    assert "timeout=\"15\"" in xml


def test_connect_action_transfers_when_requested() -> None:
    xml = twiml.connect_action_twiml(
        transfer_requested=True, number="+15559876543", message=""
    )
    assert "<Dial" in xml
    assert "<Number>+15559876543</Number>" in xml
    # empty message must not emit an empty <Say>
    assert "<Say" not in xml


def test_connect_action_without_transfer_is_empty() -> None:
    xml = twiml.connect_action_twiml(
        transfer_requested=False, number="+15559876543", message=""
    )
    assert "<Dial" not in xml
    assert "<Say" not in xml
    assert "</Response>" in xml


def test_connect_action_ignores_transfer_without_number() -> None:
    xml = twiml.connect_action_twiml(transfer_requested=True, number="", message="")
    assert "<Dial" not in xml
