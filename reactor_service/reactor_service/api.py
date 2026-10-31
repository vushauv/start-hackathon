"""Public interface: plain functions you call. Async twins are prefixed with ``a``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Coroutine, TypeVar

from . import client
from .config import ReactorSettings, get_settings
from .errors import ReactorError
from .models import Answer, AskRequest, Evidence, History, TextReply
from .protocol import build_message as _build_message
from .protocol import citations, history_part, is_abstention, new_ref

T = TypeVar("T")

EvidenceLike = Evidence | dict[str, Any]


def _settings(s: ReactorSettings | None) -> ReactorSettings:
    return s or get_settings()


def _evidence(items: Iterable[EvidenceLike]) -> list[Evidence]:
    return [e if isinstance(e, Evidence) else Evidence.model_validate(e) for e in items]


def _run(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise ReactorError(
        "called a sync reactor_service function from inside a running event loop; "
        "use the async variant (anew_history, aask, asend_text, aload_history)"
    )


# -- async core ---------------------------------------------------------------


async def anew_history(*, settings: ReactorSettings | None = None) -> History:
    return await client.new_history(_settings(settings))


async def aask(
    question: str,
    evidence: Iterable[EvidenceLike] = (),
    *,
    question_date: str | None = None,
    history_id: str | None = None,
    question_id: str | None = None,
    timeout: float | None = None,
    settings: ReactorSettings | None = None,
) -> Answer:
    s = _settings(settings)
    req = AskRequest(
        question=question,
        evidence=_evidence(evidence),
        question_date=question_date,
        history_id=history_id,
        question_id=question_id,
    )
    ref = new_ref()
    text = _build_message(req.question, req.evidence, ref=ref, question_date=req.question_date)
    raw = await client.send(s, text, ref=ref, history_id=req.history_id, timeout=timeout)
    answer = raw.answer or ""
    abstained = is_abstention(answer)
    cited, unknown = citations(answer, req.evidence)
    return Answer(
        answer=answer,
        abstained=abstained,
        citations=cited,
        unknown_citations=unknown,
        uncited=bool(req.evidence) and not cited and not abstained,
        reply=raw.reply,
        cycle=raw.cycle,
        latency_s=raw.latency_s,
        retries=raw.retries,
        other_replies=raw.other_replies,
        late_replies=raw.late_replies,
        ref=ref,
        question=req.question,
        question_date=req.question_date,
        question_id=req.question_id,
        history_id=req.history_id,
    )


async def asend_text(
    text: str,
    *,
    history_id: str | None = None,
    timeout: float | None = None,
    settings: ReactorSettings | None = None,
) -> TextReply:
    s = _settings(settings)
    raw = await client.send(s, text, ref=None, history_id=history_id, timeout=timeout)
    return TextReply(
        reply=raw.reply,
        cycle=raw.cycle,
        latency_s=raw.latency_s,
        retries=raw.retries,
        other_replies=raw.other_replies,
        late_replies=raw.late_replies,
        history_id=history_id,
    )


async def aload_history(
    parts: Sequence[str],
    *,
    history_id: str,
    timeout: float | None = None,
    settings: ReactorSettings | None = None,
    on_part: Callable[[int, int, TextReply], None] | None = None,
) -> list[TextReply]:
    """Feed a whole dataset history into Reactor memory, one part at a time.

    Each part is wrapped with "Part i of n ... Reply only OK." and the next part
    is sent only after a reply arrives. ``on_part(i, n, reply)`` is called after
    each part, handy for progress output.
    """
    replies: list[TextReply] = []
    n = len(parts)
    for i, chunk in enumerate(parts, start=1):
        r = await asend_text(
            history_part(chunk, i, n), history_id=history_id, timeout=timeout, settings=settings
        )
        replies.append(r)
        if on_part:
            on_part(i, n, r)
    return replies


# -- sync wrappers --------------------------------------------------------------


def new_history(*, settings: ReactorSettings | None = None) -> History:
    """Start a fresh Reactor history and return its id. One per dataset history."""
    return _run(anew_history(settings=settings))


def ask(
    question: str,
    evidence: Iterable[EvidenceLike] = (),
    *,
    question_date: str | None = None,
    history_id: str | None = None,
    question_id: str | None = None,
    timeout: float | None = None,
    settings: ReactorSettings | None = None,
) -> Answer:
    """Ask one question with evidence and wait for the final, cited answer."""
    return _run(
        aask(
            question,
            evidence,
            question_date=question_date,
            history_id=history_id,
            question_id=question_id,
            timeout=timeout,
            settings=settings,
        )
    )


def send_text(
    text: str,
    *,
    history_id: str | None = None,
    timeout: float | None = None,
    settings: ReactorSettings | None = None,
) -> TextReply:
    """Send free text and return the first reply (no final-answer parsing)."""
    return _run(asend_text(text, history_id=history_id, timeout=timeout, settings=settings))


def load_history(
    parts: Sequence[str],
    *,
    history_id: str,
    timeout: float | None = None,
    settings: ReactorSettings | None = None,
    on_part: Callable[[int, int, TextReply], None] | None = None,
) -> list[TextReply]:
    """Load a dataset history into Reactor memory, part by part."""
    return _run(
        aload_history(
            parts, history_id=history_id, timeout=timeout, settings=settings, on_part=on_part
        )
    )


def build_message(
    question: str,
    evidence: Iterable[EvidenceLike] = (),
    *,
    question_date: str | None = None,
    ref: str = "00000000",
) -> str:
    """The exact text :func:`ask` would send. Useful to check sizes before asking."""
    return _build_message(question, _evidence(evidence), ref=ref, question_date=question_date)


def split_history(
    sessions: Iterable[dict[str, Any]],
    *,
    max_chars: int | None = None,
    settings: ReactorSettings | None = None,
) -> list[str]:
    """Render dataset sessions (``session_id``, ``date``, ``turns``) into text parts
    that fit the message limit, oldest first. Overhead for the part wrapper is reserved.
    """
    limit = (max_chars or _settings(settings).max_chars) - 200
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for s in sessions:
        header = f"## session {s.get('session_id', '?')} ({s.get('date', '?')})"
        lines = [header]
        for t in s.get("turns", []):
            lines.append(f"{t.get('role', 'user')}: {t.get('content', '')}")
        block = "\n".join(lines)
        # split a single oversized session
        chunks = [block[i : i + limit] for i in range(0, len(block), limit)] or [""]
        for chunk in chunks:
            if size + len(chunk) + 2 > limit and buf:
                parts.append("\n\n".join(buf))
                buf, size = [], 0
            buf.append(chunk)
            size += len(chunk) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts
