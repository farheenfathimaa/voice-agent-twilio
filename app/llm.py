from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .config import Settings
from .prompt import HANDOFF_MARKER, export_env_keys
from .sessions import ToolRecord

logger = logging.getLogger("voice.llm")

Executor = Callable[[str, dict[str, Any]], Awaitable[tuple[str, ToolRecord]]]

HANDOFF_REPLY = "Of course. Let me connect you with a specialist."


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    reply: str
    model: str
    latency_ms: float = 0.0
    tool_ms: float = 0.0
    tool_calls: list[ToolRecord] = field(default_factory=list)
    rounds: int = 1
    handoff_marker: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    @property
    def uses_tools(self) -> bool:
        return bool(self.tool_calls)


class LiteLLMClient:
    name = "litellm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._litellm: Any = None

    def _module(self) -> Any:
        if self._litellm is None:
            try:
                import litellm
            except ImportError as exc:  # pragma: no cover - dependency is required in prod
                raise LLMError("litellm is not installed") from exc
            export_env_keys(self.settings)
            litellm.drop_params = True
            litellm.suppress_debug_info = True
            self._litellm = litellm
        return self._litellm

    async def _complete(self, kwargs: dict[str, Any]) -> Any:
        litellm = self._module()
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await litellm.acompletion(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("llm attempt %d failed: %s", attempt + 1, exc)
                if attempt == 0:
                    await asyncio.sleep(0.4)
        raise LLMError(str(last_exc)) from last_exc

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        executor: Executor | None = None,
    ) -> LLMResult:
        settings = self.settings
        started = time.perf_counter()
        tool_ms = 0.0
        records: list[ToolRecord] = []
        msgs = list(messages)
        rounds = 0
        last_message: Any = None

        while True:
            rounds += 1
            kwargs: dict[str, Any] = {
                "model": settings.llm_model,
                "messages": msgs,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "timeout": settings.llm_timeout_s,
                "drop_params": True,
            }
            if tools and executor is not None and rounds <= settings.max_tool_iterations:
                kwargs["tools"] = tools
            try:
                response = await self._complete(kwargs)
            except LLMError as exc:
                return LLMResult(
                    reply="",
                    model=settings.llm_model,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    tool_ms=tool_ms,
                    tool_calls=records,
                    rounds=rounds,
                    ok=False,
                    error=str(exc),
                )

            message = response.choices[0].message
            last_message = message
            calls = list(getattr(message, "tool_calls", None) or [])
            if not calls:
                break
            if executor is None:
                break

            tool_calls_payload = []
            for call in calls:
                function = getattr(call, "function", None)
                name = getattr(function, "name", "") or ""
                raw_args = getattr(function, "arguments", "{}") or "{}"
                try:
                    if isinstance(raw_args, str):
                        arguments = json.loads(raw_args)
                    else:
                        arguments = dict(raw_args)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls_payload.append(
                    {
                        "id": getattr(call, "id", "") or f"call_{len(records)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                )
                content, record = await executor(name, arguments)
                records.append(record)
                tool_ms += record.latency_ms
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls_payload[-1]["id"],
                        "name": name,
                        "content": content,
                    }
                )
            msgs.append(
                {
                    "role": "assistant",
                    "content": getattr(message, "content", None) or "",
                    "tool_calls": tool_calls_payload,
                }
            )

        reply = (getattr(last_message, "content", None) or "").strip() if last_message else ""
        handoff = HANDOFF_MARKER in reply
        if handoff:
            reply = reply.replace(HANDOFF_MARKER, "").strip()
        usage: dict[str, Any] = {}
        raw_usage = getattr(response, "usage", None) if last_message is not None else None
        if raw_usage is not None:
            try:
                usage = {
                    k: v for k, v in dict(raw_usage).items() if isinstance(v, (int, float, str))
                }
            except (TypeError, ValueError):
                usage = {}
        if not reply and records:
            reply = HANDOFF_REPLY
        return LLMResult(
            reply=reply,
            model=settings.llm_model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            tool_ms=tool_ms,
            tool_calls=records,
            rounds=rounds,
            handoff_marker=handoff,
            usage=usage,
            ok=bool(reply),
            error=None if reply else "empty reply",
        )


class MockLLM:
    """Deterministic offline LLM: tool intents + retrieval-grounded replies."""

    name = "mock"
    model = "mock-llm"

    _ORDER_RE = re.compile(r"\bord[-\s]?(\d{3,5})\b", re.IGNORECASE)
    _ORDER_QUESTIONS = re.compile(
        r"\b(where|status|track|tracking|arrive|arrival|shipping|shipped|deliver|delivery|update)\b",
        re.IGNORECASE,
    )
    _TICKET_RE = re.compile(
        r"\b(ticket|complaint|complain|escalat|follow[- ]?up|support request)\b", re.I
    )
    _CUSTOMER_RE = re.compile(
        r"\b(what(?:'s| is) my (?:tier|loyalty|membership|vip status)|"
        r"(?:look up|pull up|show) my (?:customer )?profile|my customer record)\b",
        re.I,
    )
    _NO_CONTEXT = "(no matching knowledge found)"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls: list[str] = []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        executor: Executor | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        ) or ""
        user_text = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        ) or ""
        self.calls.append(user_text)
        from .prompt import extract_context  # local import avoids cycle at module load

        context = extract_context(system)
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        records: list[ToolRecord] = []
        tool_ms = 0.0
        rounds = 0

        while rounds <= self.settings.max_tool_iterations:
            rounds += 1
            if tool_messages:
                reply = self._reply_from_tool(tool_messages[-1])
                return LLMResult(
                    reply=reply,
                    model=self.model,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    tool_ms=tool_ms,
                    tool_calls=records,
                    rounds=rounds,
                )
            intent = self._tool_intent(user_text)
            if intent is None or executor is None:
                break
            content, record = await executor(intent[0], intent[1])
            records.append(record)
            tool_ms += record.latency_ms
            tool_messages.append(
                {"role": "tool", "tool_call_id": "mock", "name": intent[0], "content": content}
            )
            messages = list(messages) + [
                {"role": "assistant", "content": "", "tool_calls": []},
                tool_messages[-1],
            ]

        reply = self._reply(user_text, context)
        handoff = HANDOFF_MARKER in reply
        if handoff:
            reply = reply.replace(HANDOFF_MARKER, "").strip()
        return LLMResult(
            reply=reply,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            tool_ms=tool_ms,
            tool_calls=records,
            rounds=rounds,
            handoff_marker=handoff,
            ok=bool(reply),
        )

    def _tool_intent(self, text: str) -> tuple[str, dict[str, Any]] | None:
        order_match = self._ORDER_RE.search(text)
        if order_match and (self._ORDER_QUESTIONS.search(text) or "ord" in text.lower()):
            return "lookup_order", {"order_id": f"ORD-{order_match.group(1)}"}
        if self._TICKET_RE.search(text):
            return "create_ticket", {
                "subject": text[:60] or "Support request",
                "description": text,
                "priority": "normal",
            }
        if self._CUSTOMER_RE.search(text):
            return "get_customer", {}
        return None

    def _reply_from_tool(self, tool_message: dict[str, Any]) -> str:
        name = tool_message.get("name", "")
        raw = tool_message.get("content", "")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = {}
        if not isinstance(data, dict):
            data = {"result": data}
        if data.get("ok") is False or "error" in data:
            error = str(data.get("error", "that did not work"))
            return (
                f"I could not pull that up: {error}. "
                f"I can open a ticket for you instead. {HANDOFF_MARKER}"
            )
        if name == "lookup_order":
            status = str(data.get("status", "unknown"))
            eta = data.get("estimated_delivery") or data.get("delivered_on") or ""
            total = data.get("total")
            bits = [f"Your order {data.get('order_id', '')} is {status}"]
            if eta:
                bits.append(f"with an estimate of {eta}")
            if total is not None:
                bits.append(f"totaling {total}")
            return ". ".join(bits) + "."
        if name == "list_orders":
            orders = data.get("orders") or []
            if not orders:
                return f"I could not find any orders on {data.get('phone', 'that number')}."
            first = orders[0]
            return (
                f"I found {len(orders)} orders. The latest, {first.get('order_id', '')}, "
                f"is {first.get('status', 'unknown')}."
            )
        if name == "get_customer":
            return (
                f"I have you as {data.get('name', 'a valued customer')}, "
                f"tier {data.get('tier', 'standard')}."
            )
        if name == "create_ticket":
            return (
                f"I have created ticket {data.get('ticket_id', '')} with priority "
                f"{data.get('priority', 'normal')}. Our team will follow up by email."
            )
        return "Done."

    def _reply(self, user_text: str, context: str) -> str:
        from .confidence import is_smalltalk  # local import avoids cycle at module load

        if is_smalltalk(user_text):
            lowered = user_text.lower()
            if any(w in lowered for w in ("bye", "goodbye")):
                return "Thanks for calling. Goodbye."
            if any(w in lowered for w in ("thank", "thanks")):
                return "You are very welcome. Is there anything else I can help with?"
            if any(w in lowered for w in ("hi", "hello", "hey")):
                return "Hello! How can I help you today?"
            return "Sure, happy to help."
        if context.strip() and not context.strip().startswith(self._NO_CONTEXT):
            block = context.strip().split("\n\n")[0]
            block = re.sub(r"^\[\d+\]\s*\([^)]*\)\s*", "", block)
            sentences = re.split(r"(?<=[.!?])\s+", block)
            snippet = " ".join(sentences[:2]).strip()
            return f"Here is what I found: {snippet}"
        return f"I do not have that information. {HANDOFF_MARKER}"


def build_llm(settings: Settings) -> Any:
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockLLM(settings)
    if provider == "litellm":
        if not (settings.groq_api_key or settings.google_api_key or settings.openai_api_key):
            logger.warning("no LLM API keys configured; falling back to MockLLM")
            return MockLLM(settings)
        return LiteLLMClient(settings)
    raise ValueError(f"unknown llm provider: {provider}")
