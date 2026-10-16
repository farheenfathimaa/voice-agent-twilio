from __future__ import annotations

import json

import pytest

from app.crm_client import CRMClient, ToolContext
from app.tools import TOOL_NAMES, TOOL_SPECS, ToolExecutor


@pytest.fixture()
def crm(settings):
    from mock_crm.server import app as crm_app

    client = CRMClient(settings)
    client.use_asgi(crm_app)
    return client


@pytest.fixture()
def executor(crm) -> ToolExecutor:
    return ToolExecutor(crm)


@pytest.fixture()
def ctx() -> ToolContext:
    return ToolContext(
        caller_id="+15550001111",
        call_sid="CAtooltest01",
        session_id="CAtooltest01",
        called="+15551230000",
    )


def test_tool_specs_expose_names() -> None:
    assert TOOL_NAMES == ["lookup_order", "list_orders", "get_customer", "create_ticket"]
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        assert spec["function"]["name"] in TOOL_NAMES


async def test_lookup_order_success(executor: ToolExecutor, ctx: ToolContext) -> None:
    content, record = await executor.execute("lookup_order", {"order_id": "ORD-1001"}, ctx)
    assert record.ok is True
    assert record.error is None
    payload = json.loads(content)
    assert payload["status"] == "shipped"
    assert payload["metadata"]["caller_id"] == "+15550001111"


async def test_lookup_order_missing_id_fails(executor: ToolExecutor, ctx: ToolContext) -> None:
    content, record = await executor.execute("lookup_order", {}, ctx)
    assert record.ok is False
    assert "order_id is required" in record.error
    assert json.loads(content)["ok"] is False


async def test_lookup_unknown_order_fails(executor: ToolExecutor, ctx: ToolContext) -> None:
    _, record = await executor.execute("lookup_order", {"order_id": "ORD-9999"}, ctx)
    assert record.ok is False
    assert "not found" in record.error.lower()


async def test_list_orders_uses_caller_id(executor: ToolExecutor, ctx: ToolContext) -> None:
    content, record = await executor.execute("list_orders", {}, ctx)
    assert record.ok is True
    payload = json.loads(content)
    assert payload["count"] == 2
    assert payload["metadata"]["caller_id"] == "+15550001111"


async def test_list_orders_without_caller_id_fails(executor: ToolExecutor) -> None:
    _, record = await executor.execute("list_orders", {}, ToolContext())
    assert record.ok is False
    assert "caller id" in record.error.lower()


async def test_get_customer(executor: ToolExecutor, ctx: ToolContext) -> None:
    content, record = await executor.execute("get_customer", {}, ctx)
    assert record.ok is True
    payload = json.loads(content)
    assert payload["name"] == "Alice Nguyen"
    assert payload["tier"] == "gold"


async def test_create_ticket(executor: ToolExecutor, ctx: ToolContext) -> None:
    content, record = await executor.execute(
        "create_ticket",
        {"subject": "Damaged item", "description": "Screen cracked", "priority": "high"},
        ctx,
    )
    assert record.ok is True
    payload = json.loads(content)
    assert payload["ticket_id"].startswith("TKT-")
    assert payload["priority"] == "high"
    assert payload["caller_id"] == "+15550001111"
    assert payload["call_sid"] == "CAtooltest01"


async def test_unknown_tool_fails(executor: ToolExecutor, ctx: ToolContext) -> None:
    _, record = await executor.execute("teleport", {}, ctx)
    assert record.ok is False
    assert "unknown tool" in record.error


async def test_crm_metadata_headers_reach_backend(executor: ToolExecutor, ctx: ToolContext) -> None:
    from mock_crm.server import REQUEST_LOG

    await executor.execute("get_customer", {}, ctx)
    assert REQUEST_LOG
    last = REQUEST_LOG[-1]
    assert last["metadata"]["caller_id"] == "+15550001111"
    assert last["metadata"]["call_sid"] == "CAtooltest01"
