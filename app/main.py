from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel, Field

from . import twiml
from .bootstrap import Components, build_components
from .config import Settings, get_settings
from .rag import RAGEngine
from .streams import MediaConnection

logger = logging.getLogger("voice.app")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    call_sid: str | None = None
    caller_id: str = ""
    called: str = ""
    synthesize: bool = True
    final: bool = False
    include_audio: bool = False


class SessionRequest(BaseModel):
    caller_id: str = ""
    call_sid: str | None = None
    called: str = ""


class TurnResponse(BaseModel):
    session_id: str
    call_sid: str
    turn: int
    reply: str
    confidence: float
    handoff: bool
    handoff_reason: str
    contained: bool
    tools: list[dict[str, Any]]
    rag_hits: list[dict[str, Any]]
    latency: dict[str, float]
    audio_b64: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        components = await build_components(settings)
        app.state.c = components
        app.state.settings = settings

        async def cleanup() -> None:
            await components.sessions.cleanup_expired()

        cleaner = asyncio.create_task(_periodic(cleanup, settings.session_cleanup_interval_s))
        logger.info("%s started (%s)", settings.app_name, settings.environment)
        try:
            yield
        finally:
            cleaner.cancel()
            with suppress(asyncio.CancelledError):
                await cleaner
            await components.shutdown()
            logger.info("%s stopped", settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Twilio Media Streams voice support agent (STT -> LLM -> TTS + RAG + CRM tools)"
        ),
        lifespan=lifespan,
    )

    def components() -> Components:
        c: Components | None = getattr(app.state, "c", None)
        if c is None:
            raise HTTPException(status_code=503, detail="components not ready")
        return c

    # ------------------------------------------------------------- telephony
    @app.post("/voice")
    async def voice_webhook(request: Request) -> Response:
        """Inbound call webhook: greet, then hand the call to the bidirectional stream."""
        form = await request.form()
        call_sid = str(form.get("CallSid") or "")
        caller_id = str(form.get("From") or "")
        called = str(form.get("To") or "")
        c = components()
        session = c.sessions.get_or_create(
            call_sid=call_sid or None, caller_id=caller_id, called=called, source="stream"
        )
        base = settings.resolved_public_base() or str(request.base_url).rstrip("/")
        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        xml = twiml.response(
            twiml.connect_stream(
                f"{ws_base}/media",
                action_url=f"{base}/voice/connect",
                parameters={
                    "caller_id": caller_id,
                    "call_sid": session.call_sid,
                    "called": called,
                },
            )
        )
        logger.info("incoming call sid=%s from=%s", session.call_sid, caller_id)
        return Response(content=xml, media_type="application/xml")

    @app.post("/voice/connect")
    async def connect_action(request: Request) -> Response:
        """TwiML executed when the <Connect> block ends (our WebSocket closed)."""
        form = await request.form()
        call_sid = str(form.get("CallSid") or "")
        c = components()
        session = c.sessions.get(call_sid)
        requested = bool(session and session.transfer_requested)
        number = settings.transfer_number
        if requested and not number:
            logger.error(
                "transfer requested for %s but TRANSFER_NUMBER is not configured", call_sid
            )
        xml = twiml.connect_action_twiml(
            transfer_requested=requested and bool(number),
            number=number,
            message="",
            caller_id=settings.twilio_phone_number or None,
            timeout=settings.dial_timeout_s,
        )
        if requested:
            logger.info("transferring call %s to %s", call_sid, number or "<unset>")
        return Response(content=xml, media_type="application/xml")

    @app.websocket("/media")
    async def media_stream(websocket: WebSocket) -> None:
        conn = MediaConnection(websocket, settings, components())
        await conn.run()

    # ----------------------------------------------------------- agent HTTP
    @app.post("/agent/sessions", status_code=201)
    async def create_session(body: SessionRequest) -> dict[str, Any]:
        c = components()
        session = c.sessions.create(caller_id=body.caller_id, call_sid=body.call_sid,
                                    called=body.called, source="http")
        return {
            "session_id": session.session_id,
            "call_sid": session.call_sid,
            "caller_id": session.caller_id,
            "greeting": settings.greeting,
        }

    @app.post("/agent/turn", response_model=TurnResponse)
    async def agent_turn(body: TurnRequest) -> TurnResponse:
        c = components()
        session = c.sessions.get_or_create(
            call_sid=body.call_sid,
            session_id=body.session_id,
            caller_id=body.caller_id,
            called=body.called,
            source="http",
        )
        outcome = await c.agent.handle_text_turn(
            body.text, session, synthesize=body.synthesize
        )
        metrics = outcome.metrics
        response = TurnResponse(
            session_id=session.session_id,
            call_sid=session.call_sid,
            turn=metrics.turn_index,
            reply=outcome.reply,
            confidence=outcome.confidence,
            handoff=outcome.handoff,
            handoff_reason=outcome.thought.handoff_reason,
            contained=not session.transfer_requested,
            tools=[t.as_dict() for t in outcome.thought.tool_records],
            rag_hits=[h.as_dict() for h in outcome.thought.hits],
            latency={
                "vad_ms": metrics.vad_ms,
                "stt_ms": metrics.stt_ms,
                "retrieve_ms": metrics.retrieve_ms,
                "llm_ms": metrics.llm_ms,
                "tool_ms": metrics.tool_ms,
                "tts_ms": metrics.tts_ms,
                "total_ms": metrics.total_ms,
            },
            audio_b64=_b64(outcome.audio) if body.include_audio else None,
        )
        if body.final:
            await c.sessions.finalize(session, reason="final_turn")
        return response

    @app.post("/calls/{call_sid}/end")
    async def end_call(call_sid: str) -> dict[str, Any]:
        c = components()
        session = c.sessions.get(call_sid)
        if session is None:
            raise HTTPException(status_code=404, detail="call not found")
        await c.sessions.finalize(session, reason="manual")
        return {"call_sid": call_sid, "summary_posted": session.summary_posted,
                "turns": len(session.turns), "handoff": session.transfer_requested}

    # --------------------------------------------------------------- tooling
    @app.post("/rag/reindex", status_code=202)
    async def reindex() -> dict[str, Any]:
        c = components()
        c.rag = await asyncio.to_thread(RAGEngine.build, settings)
        c.agent.rag = c.rag
        return c.rag.stats()

    @app.get("/rag/stats")
    async def rag_stats() -> dict[str, Any]:
        return components().rag.stats()

    @app.get("/metrics/turns")
    async def metrics_turns(limit: int = 100) -> dict[str, Any]:
        c = components()
        return {
            "summary": c.metrics.summary(),
            "turns": [t.as_dict() for t in c.metrics.recent(limit)],
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        c = components()
        return {
            "status": "ok",
            "app": settings.app_name,
            "environment": settings.environment,
            "providers": c.provider_info(),
            "rag_ready": c.rag.ready,
            "sessions": len(c.sessions.all()),
            "transfer_configured": bool(settings.transfer_number),
            "n8n_configured": bool(settings.n8n_webhook_url),
        }

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "app": settings.app_name,
            "endpoints": [
                "POST /voice",
                "POST /voice/connect",
                "WS /media",
                "POST /agent/sessions",
                "POST /agent/turn",
                "POST /calls/{call_sid}/end",
                "GET /metrics/turns",
                "GET /rag/stats",
                "GET /healthz",
            ],
        }

    return app


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii") if data else ""


async def _periodic(coro_factory: Any, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await coro_factory()
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:  # noqa: BLE001
            logger.exception("periodic task failed")


app = create_app()
