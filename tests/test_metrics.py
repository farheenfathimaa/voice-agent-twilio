from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["providers"]["embeddings"] == "hashing"
    assert body["rag_ready"] is True
    assert body["transfer_configured"] is True
    assert body["n8n_configured"] is True


def test_root_lists_endpoints(client: TestClient) -> None:
    body = client.get("/").json()
    assert "POST /voice" in body["endpoints"]
    assert "WS /media" in body["endpoints"]


def test_rag_stats(client: TestClient) -> None:
    body = client.get("/rag/stats").json()
    assert body["provider"] == "hashing"
    assert body["chunks"] > 0


def test_metrics_reflect_completed_turns(client: TestClient, turn, new_session) -> None:
    session = new_session()
    turn("How long does standard shipping take?", session)
    turn("What is your return policy?", session)

    body = client.get("/metrics/turns").json()
    assert body["summary"]["turns"] == 2
    assert "avg_total_ms" in body["summary"]
    assert len(body["turns"]) == 2
    first = body["turns"][0]
    assert first["call_sid"] == session["call_sid"]
    assert first["turn_index"] == 1
    assert first["stt_provider"] == "mock"


def test_reindex_endpoint(client: TestClient) -> None:
    resp = client.post("/rag/reindex")
    assert resp.status_code == 202
    assert resp.json()["ready"] is True
