from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings

logger = logging.getLogger("voice.crm")


@dataclass
class ToolContext:
    """Call metadata attached to every tool call / CRM request."""

    caller_id: str = ""
    call_sid: str = ""
    session_id: str = ""
    called: str = ""
    channel: str = "voice"
    extra: dict[str, str] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        headers = {
            "X-Caller-ID": self.caller_id,
            "X-Call-SID": self.call_sid,
            "X-Session-ID": self.session_id,
            "X-Channel": self.channel,
        }
        headers.update(self.extra)
        return {k: v for k, v in headers.items() if v}

    def as_dict(self) -> dict[str, str]:
        return {
            "caller_id": self.caller_id,
            "call_sid": self.call_sid,
            "session_id": self.session_id,
            "called": self.called,
            "channel": self.channel,
        }


@dataclass
class CRMResult:
    ok: bool
    status: int = 0
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class CRMClient:
    """Thin REST client for the mock CRM. Never raises."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.crm_base_url.rstrip("/"),
            timeout=settings.crm_timeout_s,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def use_asgi(self, asgi_app: Any, base_url: str = "http://mock-crm.local") -> None:
        """Route CRM calls into an in-process ASGI app (tests / offline eval)."""
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app),
            base_url=base_url,
            timeout=self.settings.crm_timeout_s,
        )
        self._owns_client = True

    def _headers(self, ctx: ToolContext) -> dict[str, str]:
        headers = ctx.headers()
        if self.settings.crm_api_key:
            headers["X-API-Key"] = self.settings.crm_api_key
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        ctx: ToolContext,
        json_body: dict[str, Any] | None = None,
    ) -> CRMResult:
        started = time.perf_counter()
        try:
            resp = await self._client.request(
                method, path, headers=self._headers(ctx), json=json_body
            )
        except httpx.HTTPError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            logger.warning("crm %s %s failed: %s", method, path, exc)
            return CRMResult(ok=False, error=str(exc), latency_ms=latency)
        latency = (time.perf_counter() - started) * 1000.0
        if resp.status_code >= 400:
            detail = resp.text[:200]
            try:
                payload = resp.json()
                detail = str(payload.get("detail") or payload.get("error") or detail)
            except (ValueError, AttributeError):
                pass
            return CRMResult(
                ok=False, status=resp.status_code, error=detail, latency_ms=latency
            )
        try:
            data: Any = resp.json()
        except ValueError:
            data = {"raw": resp.text}
        return CRMResult(ok=True, status=resp.status_code, data=data, latency_ms=latency)

    async def get_order(self, order_id: str, ctx: ToolContext) -> CRMResult:
        return await self._request("GET", f"/v1/orders/{order_id.strip()}", ctx)

    async def list_orders(self, phone: str, ctx: ToolContext) -> CRMResult:
        return await self._list_orders(phone, ctx)

    async def _list_orders(self, phone: str, ctx: ToolContext) -> CRMResult:
        started = time.perf_counter()
        try:
            resp = await self._client.get(
                "/v1/orders",
                params={"phone": phone},
                headers=self._headers(ctx),
            )
        except httpx.HTTPError as exc:
            return CRMResult(
                ok=False, error=str(exc), latency_ms=(time.perf_counter() - started) * 1000.0
            )
        latency = (time.perf_counter() - started) * 1000.0
        if resp.status_code >= 400:
            return CRMResult(
                ok=False, status=resp.status_code, error=resp.text[:200], latency_ms=latency
            )
        try:
            return CRMResult(ok=True, status=resp.status_code, data=resp.json(), latency_ms=latency)
        except ValueError:
            return CRMResult(ok=True, status=resp.status_code, data={}, latency_ms=latency)

    async def get_customer(self, phone: str, ctx: ToolContext) -> CRMResult:
        return await self._request("GET", f"/v1/customers/{phone.strip()}", ctx)

    async def create_ticket(
        self,
        subject: str,
        description: str,
        priority: str,
        ctx: ToolContext,
    ) -> CRMResult:
        body = {
            "subject": subject,
            "description": description,
            "priority": priority,
            "metadata": ctx.as_dict(),
        }
        return await self._request("POST", "/v1/tickets", ctx, json_body=body)

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/healthz")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
