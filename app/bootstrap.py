from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .agent import VoiceAgent
from .config import Settings
from .crm_client import CRMClient
from .latency import MetricsStore
from .llm import build_llm
from .notifier import SummaryNotifier
from .prompt import export_env_keys
from .rag import RAGEngine
from .sessions import CallSession, SessionRegistry
from .stt import build_stt
from .tools import ToolExecutor
from .tts import build_tts

logger = logging.getLogger("voice.bootstrap")


async def _make_end_handler(notifier: SummaryNotifier, settings: Settings):
    async def handler(session: CallSession) -> None:
        if not session.turns and not session.transfer_requested:
            logger.debug("skipping summary for empty call %s", session.call_sid)
            return
        await notifier.notify(session)

    return handler


@dataclass
class Components:
    settings: Settings
    rag: RAGEngine
    stt: Any
    tts: Any
    llm: Any
    crm: CRMClient
    tools: ToolExecutor
    agent: VoiceAgent
    sessions: SessionRegistry
    metrics: MetricsStore
    notifier: SummaryNotifier

    def provider_info(self) -> dict[str, str]:
        return {
            "llm": f"{self.settings.llm_provider}/{getattr(self.llm, 'name', '?')}",
            "stt": str(getattr(self.stt, "name", "?")),
            "tts": str(getattr(self.tts, "name", "?")),
            "embeddings": self.rag.provider,
            "vector_backend": self.rag.index_state.get("backend", "n/a"),
        }

    async def shutdown(self) -> None:
        closers = [self.stt, self.tts, self.crm, self.notifier]
        for closer in closers:
            close = getattr(closer, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:  # noqa: BLE001
                logger.exception("error closing %s", type(closer).__name__)


async def build_components(settings: Settings) -> Components:
    export_env_keys(settings)
    rag = await asyncio.to_thread(RAGEngine.build, settings)
    stt = build_stt(settings)
    tts = build_tts(settings)
    llm = build_llm(settings)
    crm = CRMClient(settings)
    tools = ToolExecutor(crm)
    notifier = SummaryNotifier(settings)
    sessions = SessionRegistry(ttl_s=settings.session_ttl_s)
    sessions.set_end_handler(await _make_end_handler(notifier, settings))
    metrics = MetricsStore(window=settings.metrics_window)
    agent = VoiceAgent(settings=settings, rag=rag, llm=llm, stt=stt, tts=tts, tools=tools,
                       metrics=metrics)
    logger.info("components ready: %s", provider_summary(settings, rag))
    return Components(
        settings=settings,
        rag=rag,
        stt=stt,
        tts=tts,
        llm=llm,
        crm=crm,
        tools=tools,
        agent=agent,
        sessions=sessions,
        metrics=metrics,
        notifier=notifier,
    )


def provider_summary(settings: Settings, rag: RAGEngine) -> dict[str, Any]:
    return {
        "llm": settings.llm_model,
        "stt": settings.stt_provider,
        "tts": settings.tts_provider,
        "rag": rag.provider,
    }


async def run_periodic(coro_factory: Callable[[], Awaitable[Any]], interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await coro_factory()
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:  # noqa: BLE001
            logger.exception("periodic task failed")
