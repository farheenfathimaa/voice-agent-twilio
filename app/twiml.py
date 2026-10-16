"""Minimal TwiML builders (no twilio SDK dependency)."""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr


def _attrs(**kwargs: str) -> str:
    return "".join(f" {k.replace('_', '-')}={quoteattr(str(v))}" for k, v in kwargs.items() if v)


def response(*inner: str) -> str:
    body = "".join(inner)
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>'


def say(text: str, voice: str = "Polly.Joanna", language: str = "en-US") -> str:
    return f"<Say voice={quoteattr(voice)} language={quoteattr(language)}>{escape(text)}</Say>"


def pause(seconds: int = 1) -> str:
    return f'<Pause length="{int(seconds)}"/>'


def hangup() -> str:
    return "<Hangup/>"


def redirect(url: str) -> str:
    return f"<Redirect method=\"POST\">{escape(url)}</Redirect>"


def stream(
    ws_url: str,
    parameters: dict[str, str] | None = None,
    name: str = "voice-agent",
) -> str:
    params = ""
    if parameters:
        params = "".join(
            f"<Parameter name={quoteattr(k)} value={quoteattr(v)}/>"
            for k, v in parameters.items()
            if v is not None
        )
    return f"<Stream url={quoteattr(ws_url)} name={quoteattr(name)}>{params}</Stream>"


def connect(inner: str, action_url: str | None = None, method: str = "POST") -> str:
    action = f" action={quoteattr(action_url)} method={quoteattr(method)}" if action_url else ""
    return f"<Connect{action}>{inner}</Connect>"


def connect_stream(
    ws_url: str,
    action_url: str | None = None,
    parameters: dict[str, str] | None = None,
) -> str:
    return connect(stream(ws_url, parameters=parameters), action_url=action_url)


def dial(
    number: str,
    caller_id: str | None = None,
    timeout: int = 30,
    action: str | None = None,
    record: bool = False,
    inner: str = "",
) -> str:
    attrs = _attrs(
        timeout=str(timeout),
        callerId=caller_id or "",
        action=action or "",
        record="true" if record else "",
    )
    body = inner or f"<Number>{escape(number)}</Number>"
    return f"<Dial{attrs}>{body}</Dial>"


def transfer_twiml(
    number: str,
    message: str,
    caller_id: str | None = None,
    timeout: int = 30,
) -> str:
    inner = ""
    if message:
        inner += say(message)
    inner += dial(number, caller_id=caller_id, timeout=timeout)
    return response(inner)


def connect_action_twiml(
    transfer_requested: bool,
    number: str,
    message: str,
    caller_id: str | None = None,
    timeout: int = 30,
    no_transfer_message: str = "",
) -> str:
    """TwiML returned when the bidirectional <Connect> block ends (WebSocket closed)."""
    if transfer_requested and number:
        return transfer_twiml(number, message, caller_id=caller_id, timeout=timeout)
    parts = []
    if no_transfer_message:
        parts.append(say(no_transfer_message))
    return response("".join(parts))
