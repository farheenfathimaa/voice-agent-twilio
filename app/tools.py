from __future__ import annotations

import json
import logging
import time
from typing import Any

from .crm_client import CRMClient, ToolContext
from .sessions import ToolRecord

logger = logging.getLogger("voice.tools")

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up one order by its order id (format ORD-1001). Returns status, "
                "dates, total and tracking info for the caller."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id exactly as the caller said it, e.g. ORD-1001.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": (
                "List the orders belonging to the caller's phone number. The caller id is "
                "attached automatically; only pass phone if the caller corrects it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": (
                            "Optional phone override (E.164). Defaults to the caller id."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Fetch the caller's CRM profile (name, tier, email, open tickets).",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "Optional phone override. Defaults to the caller id.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "Create a support ticket in the CRM. Use when the caller's issue cannot be "
                "resolved on the call. Call metadata (caller id, call sid) is attached "
                "automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Short ticket subject."},
                    "description": {"type": "string", "description": "What the caller needs."},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Ticket priority.",
                    },
                },
                "required": ["subject", "description"],
            },
        },
    },
]

TOOL_NAMES = [spec["function"]["name"] for spec in TOOL_SPECS]


class ToolExecutor:
    def __init__(self, crm: CRMClient) -> None:
        self.crm = crm

    async def execute(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> tuple[str, ToolRecord]:
        started = time.perf_counter()
        arguments = arguments or {}
        data: Any = None
        error: str | None = None
        try:
            if name == "lookup_order":
                order_id = str(arguments.get("order_id") or "").strip()
                if not order_id:
                    raise ValueError("order_id is required")
                result = await self.crm.get_order(order_id, ctx)
            elif name == "list_orders":
                phone = str(arguments.get("phone") or ctx.caller_id or "").strip()
                if not phone:
                    raise ValueError("no caller id available for list_orders")
                result = await self.crm.list_orders(phone, ctx)
            elif name == "get_customer":
                phone = str(arguments.get("phone") or ctx.caller_id or "").strip()
                if not phone:
                    raise ValueError("no caller id available for get_customer")
                result = await self.crm.get_customer(phone, ctx)
            elif name == "create_ticket":
                priority = str(arguments.get("priority") or "normal").lower()
                if priority not in ("low", "normal", "high"):
                    priority = "normal"
                result = await self.crm.create_ticket(
                    subject=str(arguments.get("subject") or "Support request"),
                    description=str(arguments.get("description") or ""),
                    priority=priority,
                    ctx=ctx,
                )
            else:
                raise ValueError(f"unknown tool: {name}")
            if result.ok:
                data = result.data
            else:
                error = result.error or f"crm status {result.status}"
                data = {"ok": False, "error": error}
        except ValueError as exc:
            error = str(exc)
            data = {"ok": False, "error": error}
        except Exception as exc:  # noqa: BLE001 - tool failures must not break the turn
            logger.exception("tool %s crashed", name)
            error = str(exc)
            data = {"ok": False, "error": error}

        latency = (time.perf_counter() - started) * 1000.0
        record = ToolRecord(
            name=name,
            arguments=arguments,
            ok=error is None,
            latency_ms=latency,
            error=error,
            result=data if error is None else None,
        )
        content = json.dumps(data if data is not None else {"ok": False, "error": error})
        logger.info(
            json.dumps(
                {
                    "event": "tool_call",
                    "tool": name,
                    "ok": record.ok,
                    "latency_ms": round(latency, 1),
                    "call_sid": ctx.call_sid,
                    "caller_id": ctx.caller_id,
                    "error": error,
                }
            )
        )
        return content, record
