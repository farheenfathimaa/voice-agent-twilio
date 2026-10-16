"""Shared fixtures for the voice-agent test suite.

Everything runs offline: mock LLM/STT/TTS, the hashing embedder (no model
downloads) and the mock CRM mounted in-process over an ASGI transport.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        log_level="WARNING",
        llm_provider="mock",
        stt_provider="mock",
        tts_provider="mock",
        embeddings_provider="hashing",
        public_base_url="http://test.local",
        greeting="",
        twilio_phone_number="+15551230000",
        transfer_number="+15559876543",
        n8n_webhook_url="http://n8n.test/webhook/call",
        summarize_calls=True,
    )


@pytest.fixture()
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture()
def notifications() -> list[dict[str, Any]]:
    return []


@pytest.fixture()
def client(app, notifications: list[dict[str, Any]]):
    """TestClient with the mock CRM and a recording n8n webhook wired in."""
    from mock_crm.server import app as crm_app

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            notifications.append(json.loads(request.content.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            notifications.append({"_raw": request.content.decode("utf-8", "replace")})
        return httpx.Response(200, json={"ok": True})

    with TestClient(app) as test_client:
        components = app.state.c
        components.crm.use_asgi(crm_app)
        components.notifier._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        yield test_client


@pytest.fixture()
def components(client: TestClient):
    return client.app.state.c


@pytest.fixture()
def new_session(client: TestClient):
    def _new_session(caller_id: str = "+15550001111", **kwargs: Any) -> dict[str, Any]:
        payload = {"caller_id": caller_id, "called": "+15551230000", **kwargs}
        resp = client.post("/agent/sessions", json=payload)
        assert resp.status_code == 201, resp.text
        return resp.json()

    return _new_session


@pytest.fixture()
def turn(client: TestClient):
    def _turn(
        text: str,
        session: dict[str, Any] | None = None,
        *,
        synthesize: bool = False,
        final: bool = False,
        include_audio: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "synthesize": synthesize,
            "final": final,
            "include_audio": include_audio,
        }
        if session is not None:
            payload["session_id"] = session["session_id"]
            payload["call_sid"] = session["call_sid"]
            payload["caller_id"] = session.get("caller_id", "")
        resp = client.post("/agent/turn", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()

    return _turn
