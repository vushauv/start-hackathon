"""Answering stage: send a composed question to Reactor and get its final answer.

``ReactorAnswerer`` keeps the API's rules: one Reactor history per dataset history, one question at a time,
retries on transient failures, a fresh history when the server says ours changed. ``DryRunAnswerer`` never
connects: it is for checking message sizes and for tests of the surrounding plumbing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import reactor_service as reactor
from reactor_service import Answer, Closed, NoAnswer, ReactorSettings, Refused, TimedOut

from .models import Retrieved
from .prompting import Composed


class Answerer(Protocol):
    def start_history(self, dataset_history_id: str) -> str:
        """Begin a fresh Reactor memory for a dataset history. Returns the Reactor history id."""
        ...

    def ask(self, r: Retrieved, composed: Composed) -> tuple[Answer, int]:
        """``(answer, attempts)``. Raises ``reactor_service.NoAnswer`` when it gave up."""
        ...


class ReactorAnswerer:
    def __init__(self, settings: ReactorSettings | None = None, *, attempts: int = 3,
                 timeout: float | None = None, log: Callable[[str], None] = lambda s: None):
        self.settings = settings or reactor.get_settings()
        self.attempts = max(1, attempts)
        self.timeout = timeout
        self.log = log
        self.history_id: str | None = None

    def start_history(self, dataset_history_id: str) -> str:
        last: Exception | None = None
        for i in range(self.attempts):
            try:
                h = reactor.new_history(settings=self.settings)
                self.history_id = h.history_id
                self.log(f"[{dataset_history_id}] new Reactor history {h.history_id}")
                return h.history_id
            except reactor.ConnectFailed as e:
                last = e
                if e.status not in (409, 503):
                    raise
                self.log(f"new history refused ({e.status}), retrying: {e}")
                time.sleep(5 * (i + 1))
        assert last is not None
        raise last

    def ask(self, r: Retrieved, composed: Composed) -> tuple[Answer, int]:
        last: NoAnswer | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                a = reactor.ask(
                    r.question, [e.model_dump() for e in composed.evidence],
                    question_date=r.question_date or None, history_id=self.history_id,
                    question_id=r.question_id, notes=composed.notes,
                    timeout=self.timeout, settings=self.settings,
                )
                return a, attempt
            except Refused as e:
                last = e
                code = e.info.code
                if code == "history_changed" and e.info.current_history_id:
                    self.history_id = e.info.current_history_id
                elif code in ("history_required", "history_changed", "previous_pending"):
                    self.start_history(r.history_id)
                elif code == "history_resetting":
                    time.sleep(5)
                elif code == "too_large":
                    raise  # compose() should have prevented this; nothing a retry can do
                else:
                    time.sleep(2)
                self.log(f"{r.question_id}: refused ({code}) on attempt {attempt}: retrying")
            except (TimedOut, Closed) as e:
                last = e
                self.log(f"{r.question_id}: {type(e).__name__} on attempt {attempt}: {e}")
                if isinstance(e, Closed):
                    time.sleep(3)
        assert last is not None
        raise last


class DryRunAnswerer:
    """No network: answers with the first passage cited, so the plumbing can be exercised end to end."""

    def __init__(self):
        self.n = 0

    def start_history(self, dataset_history_id: str) -> str:
        self.n += 1
        return f"dry-{self.n}"

    def ask(self, r: Retrieved, composed: Composed) -> tuple[Answer, int]:
        from reactor_service import Citation

        cited = [Citation(label="E1", id=composed.evidence[0].id)] if composed.evidence else []
        a = Answer(answer="(dry run) [E1]" if cited else "(dry run)", abstained=False, citations=cited,
                   reply="", cycle=0, latency_s=0.0, ref="00000000", question=r.question,
                   question_date=r.question_date or None, question_id=r.question_id,
                   history_id=f"dry-{self.n}", uncited=False)
        return a, 1
