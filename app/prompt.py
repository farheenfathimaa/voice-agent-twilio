from __future__ import annotations

import os
import re
import time
from typing import Any

from .config import Settings

HANDOFF_MARKER = "CONNECTING_YOU"

CONTEXT_START = "RELEVANT KNOWLEDGE"
CONTEXT_END = "END OF KNOWLEDGE"


def build_system_prompt(
    settings: Settings,
    context: str,
    caller_id: str,
    call_sid: str,
    tool_names: list[str] | None = None,
) -> str:
    default_tools = ["lookup_order", "list_orders", "get_customer", "create_ticket"]
    tools = ", ".join(tool_names or default_tools)
    return f"""You are {settings.agent_name}, the phone support agent for {settings.business_name}.

CALL CONTEXT
- Caller ID: {caller_id or "unknown"}
- Call SID: {call_sid}
- Channel: inbound voice call
Your tools attach these identifiers automatically, so never ask the caller to repeat them.

VOICE STYLE
- Reply in 1-3 short spoken sentences. No lists, markdown, emoji, or stage directions.
- Quote only the numbers that matter (order ids, dates, amounts) exactly as given to you.
- Never invent order statuses, dates, prices, or policies.

RELEVANT KNOWLEDGE
{context or "(no matching knowledge found)"}
END OF KNOWLEDGE

TOOLS ({tools})
- Use a tool when the caller asks about their account, an order, or wants something done.
- Use at most one tool per turn, then give a plain-spoken final answer.
- If the knowledge above does not cover the question and no tool can answer, say briefly that
  you will connect the caller with a specialist and end your reply with the exact token
  {HANDOFF_MARKER}.
"""


def extract_context(system_prompt: str) -> str:
    match = re.search(
        rf"{re.escape(CONTEXT_START)}\n(.*?)\n{re.escape(CONTEXT_END)}",
        system_prompt,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def export_env_keys(settings: Settings) -> None:
    """Make provider keys visible to LiteLLM through the standard env var names."""
    mapping = {
        "GROQ_API_KEY": settings.groq_api_key,
        "GEMINI_API_KEY": settings.google_api_key,
        "OPENAI_API_KEY": settings.openai_api_key,
    }
    for name, value in mapping.items():
        if value and not os.environ.get(name):
            os.environ[name] = value
    if settings.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def as_meta(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}
