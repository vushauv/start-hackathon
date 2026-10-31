"""Async core: HTTP history creation and the WebSocket chat protocol."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import websockets
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from .config import ReactorSettings
from .errors import Closed, ConnectFailed, MessageTooLarge, Refused, TimedOut
from .models import History, _as_refusal
from .protocol import final_answers

_RETRY_START = 1.0
_RETRY_CAP = 8.0


@dataclass
class RawReply:
    """What one WebSocket exchange produced."""

    reply: str
    cycle: int
    latency_s: float
    answer: str | None = None  # the segment for our ref, when a ref was sent
    other_replies: list[str] = field(default_factory=list)
    late_replies: list[str] = field(default_factory=list)
    retries: int = 0


# -- history ------------------------------------------------------------------


async def new_history(settings: ReactorSettings, *, timeout: float | None = None) -> History:
    """``POST /agents/{agent_id}/history``: start a fresh Reactor history."""
    t = timeout or settings.timeout
    try:
        async with httpx.AsyncClient(timeout=t) as http:
            r = await http.post(settings.history_url(), headers=settings.auth_headers())
    except httpx.ConnectError as e:
        raise ConnectFailed(_dns_hint(settings, e)) from None
    except httpx.TimeoutException:
        raise TimedOut(f"new history: no response within {t}s") from None

    if r.status_code >= 400:
        detail: Any = None
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text[:200]
        raise ConnectFailed(_history_hint(r.status_code, detail), status=r.status_code)
    return History.model_validate(r.json())


# -- chat ---------------------------------------------------------------------


async def send(
    settings: ReactorSettings,
    text: str,
    *,
    ref: str | None = None,
    history_id: str | None = None,
    timeout: float | None = None,
    max_chars: int | None = None,
) -> RawReply:
    """Send one message and wait for the reply, retrying ``busy``/``previous_pending``.

    With ``ref`` the reply must contain ``# FINAL ANSWER [ref:<ref>]:``;
    without it the first reply of a newer cycle is returned.
    """
    limit = max_chars or settings.max_chars
    if len(text) > limit:
        raise MessageTooLarge(len(text), limit)

    t = timeout or settings.timeout
    deadline = time.monotonic() + t
    delay = _RETRY_START
    retries = 0
    late: list[str] = []
    try:
        async with asyncio.timeout(t):
            while True:
                try:
                    out = await _exchange(settings, text, ref=ref, history_id=history_id)
                    out.late_replies = late + out.late_replies
                    out.retries = retries
                    return out
                except Refused as e:
                    late.extend(e.late_replies)
                    if not e.retryable:
                        e.late_replies = late
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= delay:
                        e.late_replies = late
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, _RETRY_CAP)
                    retries += 1
    except TimeoutError:
        raise TimedOut(f"no answer within {t}s", late_replies=late) from None


async def _exchange(
    settings: ReactorSettings,
    text: str,
    *,
    ref: str | None,
    history_id: str | None,
) -> RawReply:
    url = settings.chat_url(history_id)
    start = time.monotonic()
    other: list[str] = []
    late: list[str] = []
    try:
        async with connect(url, additional_headers=settings.auth_headers()) as ws:
            try:
                await ws.send(json.dumps({"type": "message", "text": text}))
            except ConnectionClosed:
                pass  # a buffered refusal frame may still be readable below

            ack_cycle: int | None = None
            async for raw in ws:
                frame = _parse(raw)
                if frame is None:
                    continue
                kind = frame.get("type")

                if kind == "ack":
                    if ack_cycle is None and frame.get("text") == text:
                        ack_cycle = _int(frame.get("cycle"), 0)

                elif kind == "error":
                    raise Refused(_as_refusal(frame), other_replies=other, late_replies=late)

                elif kind == "reply":
                    reply = frame.get("text")
                    if not isinstance(reply, str):
                        continue
                    cycle = _int(frame.get("cycle"), 0)
                    if ack_cycle is not None and cycle <= ack_cycle:
                        continue  # a reply to an earlier message
                    latency = round(time.monotonic() - start, 2)
                    if ref is None:
                        return RawReply(reply, cycle, latency, other_replies=other, late_replies=late)
                    finals = final_answers(reply)
                    for k, seg in finals.items():
                        if k != ref.lower():
                            late.append(seg)
                    if ref.lower() in finals:
                        return RawReply(
                            reply, cycle, latency,
                            answer=finals[ref.lower()], other_replies=other, late_replies=late,
                        )
                    other.append(reply)
    except InvalidStatus as e:
        raise ConnectFailed(_connect_hint(e.response.status_code), status=e.response.status_code) from None
    except (socket.gaierror, OSError) as e:
        raise ConnectFailed(_dns_hint(settings, e)) from None
    except ConnectionClosed:
        raise Closed("server closed the connection", other_replies=other, late_replies=late) from None
    raise Closed("server closed the connection without an answer", other_replies=other, late_replies=late)


# -- helpers ------------------------------------------------------------------


def _parse(raw: str | bytes) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _int(v: Any, default: int) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else default


def _connect_hint(status: int) -> str:
    if status in (401, 403):
        return (
            f"refused at connect (HTTP {status}): check REACTOR_ACCESS_KEY and REACTOR_AGENT_ID; "
            "if both are right the agent is not running, ask support"
        )
    if status == 404:
        return f"refused at connect (HTTP {status}): REACTOR_BASE_URL must be just https://HOST, no path"
    return f"refused at connect (HTTP {status})"


def _history_hint(status: int, detail: Any) -> str:
    msg = f"new history refused (HTTP {status})"
    if detail:
        msg += f": {detail}"
    if status == 409:
        msg += " - a new history or maintenance is already running, retry shortly"
    elif status == 503 and detail == "still_stopping":
        msg += " - the last run is still shutting down, retry in a minute"
    elif status == 503 and detail == "restart_failed":
        msg += " - history is in place but the agent did not start; retry or ask the organizers"
    return msg


def _dns_hint(settings: ReactorSettings, e: Exception) -> str:
    return (
        f"cannot connect to {settings.base_url}: {e}. Check the host against your handout; "
        "if it is correct, ask support (the API may not be open yet)"
    )


__all__ = ["new_history", "send", "RawReply", "websockets"]
