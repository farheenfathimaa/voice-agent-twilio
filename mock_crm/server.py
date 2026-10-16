"""Mock CRM REST API used by the agent's tool layer.

Endpoints:
    GET  /healthz
    GET  /v1/customers/{phone}
    GET  /v1/orders/{order_id}
    GET  /v1/orders?phone=+15550001111
    POST /v1/tickets
    GET  /v1/tickets/{ticket_id}
    GET  /v1/_debug/requests     (observes caller metadata propagation)

Every request must carry X-API-Key plus the call metadata headers emitted by the
agent: X-Caller-ID, X-Call-SID, X-Session-ID.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

API_KEY = os.environ.get("CRM_API_KEY", "dev-crm-key")

app = FastAPI(title="mock-crm", version="0.1.0")


def require_auth(
    x_api_key: str | None = Header(default=None),
    x_caller_id: str | None = Header(default=None),
    x_call_sid: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
    x_channel: str | None = Header(default=None),
) -> dict[str, str]:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
    return {
        "caller_id": x_caller_id or "",
        "call_sid": x_call_sid or "",
        "session_id": x_session_id or "",
        "channel": x_channel or "",
    }


@dataclass
class Customer:
    phone: str
    name: str
    email: str
    tier: str = "standard"
    open_tickets: int = 0


@dataclass
class Order:
    order_id: str
    phone: str
    status: str
    total: str
    placed_on: str
    estimated_delivery: str = ""
    delivered_on: str = ""
    tracking_number: str = ""
    item: str = ""


CUSTOMERS: dict[str, Customer] = {
    "+15550001111": Customer(
        phone="+15550001111",
        name="Alice Nguyen",
        email="alice@example.com",
        tier="gold",
    ),
    "+15550002222": Customer(
        phone="+15550002222",
        name="Bob Martinez",
        email="bob@example.com",
        tier="silver",
    ),
    "+15550003333": Customer(
        phone="+15550003333",
        name="Carla Dias",
        email="carla@example.com",
        tier="standard",
    ),
}

ORDERS: dict[str, Order] = {
    "ORD-1001": Order(
        order_id="ORD-1001",
        phone="+15550001111",
        status="shipped",
        total="$49.99",
        placed_on="2026-10-08",
        estimated_delivery="2026-10-17",
        tracking_number="1Z999AA10123456784",
        item="Ergonomic keyboard",
    ),
    "ORD-1002": Order(
        order_id="ORD-1002",
        phone="+15550001111",
        status="processing",
        total="$129.00",
        placed_on="2026-10-14",
        estimated_delivery="2026-10-20",
        item="27-inch monitor",
    ),
    "ORD-2001": Order(
        order_id="ORD-2001",
        phone="+15550002222",
        status="delivered",
        total="$79.50",
        placed_on="2026-09-30",
        delivered_on="2026-10-04",
        tracking_number="1Z999AA10198765432",
        item="Wireless headset",
    ),
    "ORD-2002": Order(
        order_id="ORD-2002",
        phone="+15550002222",
        status="refunded",
        total="$24.00",
        placed_on="2026-09-12",
        delivered_on="2026-09-16",
        item="USB-C hub",
    ),
    "ORD-3001": Order(
        order_id="ORD-3001",
        phone="+15550003333",
        status="cancelled",
        total="$15.25",
        placed_on="2026-10-01",
        item="HDMI cable",
    ),
}

TICKETS: dict[str, dict[str, Any]] = {}
_ticket_seq = 1000

REQUEST_LOG: list[dict[str, Any]] = []


class TicketIn(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    priority: str = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


def _log_request(kind: str, path: str, meta: dict[str, str], payload: Any = None) -> None:
    REQUEST_LOG.append(
        {
            "ts": time.time(),
            "kind": kind,
            "path": path,
            "metadata": meta,
            "payload": payload,
        }
    )
    if len(REQUEST_LOG) > 500:
        del REQUEST_LOG[:-500]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "mock-crm"}


@app.get("/v1/customers/{phone}")
def get_customer(phone: str, meta: dict[str, str] = Depends(require_auth)) -> dict[str, Any]:
    _log_request("GET", f"/v1/customers/{phone}", meta)
    phone = phone.strip()
    customer = CUSTOMERS.get(phone)
    if customer is None:
        return {
            "phone": phone,
            "name": "Guest",
            "email": "",
            "tier": "standard",
            "open_tickets": 0,
            "known": False,
        }
    return {
        "phone": customer.phone,
        "name": customer.name,
        "email": customer.email,
        "tier": customer.tier,
        "open_tickets": customer.open_tickets,
        "known": True,
    }


@app.get("/v1/orders/{order_id}")
def get_order(order_id: str, meta: dict[str, str] = Depends(require_auth)) -> dict[str, Any]:
    _log_request("GET", f"/v1/orders/{order_id}", meta)
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise HTTPException(status_code=404, detail=f"order {order_id} not found")
    return {
        "order_id": order.order_id,
        "phone": order.phone,
        "status": order.status,
        "total": order.total,
        "placed_on": order.placed_on,
        "estimated_delivery": order.estimated_delivery,
        "delivered_on": order.delivered_on,
        "tracking_number": order.tracking_number,
        "item": order.item,
        "metadata": meta,
    }


@app.get("/v1/orders")
def list_orders(
    phone: str = Query(...), meta: dict[str, str] = Depends(require_auth)
) -> dict[str, Any]:
    _log_request("GET", "/v1/orders", meta, {"phone": phone})
    orders = [o for o in ORDERS.values() if o.phone == phone.strip()]
    return {
        "phone": phone,
        "count": len(orders),
        "orders": [
            {
                "order_id": o.order_id,
                "status": o.status,
                "total": o.total,
                "placed_on": o.placed_on,
                "item": o.item,
            }
            for o in sorted(orders, key=lambda o: o.placed_on, reverse=True)
        ],
        "metadata": meta,
    }


@app.post("/v1/tickets", status_code=201)
def create_ticket(ticket: TicketIn, meta: dict[str, str] = Depends(require_auth)) -> dict[str, Any]:
    global _ticket_seq
    _ticket_seq += 1
    ticket_id = f"TKT-{_ticket_seq}"
    if ticket.priority not in ("low", "normal", "high"):
        ticket.priority = "normal"
    record = {
        "ticket_id": ticket_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "priority": ticket.priority,
        "status": "open",
        "created_at": time.time(),
        "metadata": {**meta, **ticket.metadata},
        "caller_id": meta.get("caller_id") or ticket.metadata.get("caller_id", ""),
        "call_sid": meta.get("call_sid") or ticket.metadata.get("call_sid", ""),
    }
    TICKETS[ticket_id] = record
    _log_request("POST", "/v1/tickets", meta, ticket.model_dump())
    return record


@app.get("/v1/tickets/{ticket_id}")
def get_ticket(ticket_id: str, meta: dict[str, str] = Depends(require_auth)) -> dict[str, Any]:
    _log_request("GET", f"/v1/tickets/{ticket_id}", meta)
    ticket = TICKETS.get(ticket_id.strip().upper())
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    return ticket


@app.get("/v1/_debug/requests")
def debug_requests(meta: dict[str, str] = Depends(require_auth)) -> dict[str, Any]:
    return {"count": len(REQUEST_LOG), "requests": REQUEST_LOG[-100:]}


@app.post("/v1/_debug/reset")
def debug_reset(meta: dict[str, str] = Depends(require_auth)) -> dict[str, str]:
    REQUEST_LOG.clear()
    TICKETS.clear()
    return {"status": "reset"}
