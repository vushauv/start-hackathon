"""Answering stage: send a composed question to Reactor and get its final answer.

``ReactorAnswerer`` keeps the API's rules: one Reactor history per dataset history, one question at a time,
retries on transient failures, a fresh history when the server says ours changed. ``DryRunAnswerer`` never
connects: it is for checking message sizes and for tests of the surrounding plumbing. ``GroqAnswerer`` sends
the same message to a Groq-hosted LLM instead (``--provider groq``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx

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


# ---------------------------------------------------------------------------------------------- Groq
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Picked from GET /v1/models (Oct 2026): the strongest listed model, 131072-token context window.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_DEFAULT_MAX_CHARS = 60000
GROQ_ENV_FILE = Path(__file__).resolve().parent.parent / ".env.groq"


def groq_api_key() -> str:
    """GROQ_API_KEY from the environment, else from pipeline/.env.groq (gitignored). Never logged."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key and GROQ_ENV_FILE.exists():
        for line in GROQ_ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise reactor.ConfigError(f"GROQ_API_KEY not set (environment or {GROQ_ENV_FILE})")
    return key


def _retry_after(resp: httpx.Response, attempt: int) -> float:
    for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        v = resp.headers.get(h)
        if not v:
            continue
        try:
            return min(float(v), 120.0) + 0.5
        except ValueError:  # Groq durations like "1m26.4s" / "2.047s" / "577ms"
            import re
            m = re.fullmatch(r"(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?", v.strip())
            if m and any(m.groups()):
                s = int(m.group(1) or 0) * 60 + float(m.group(2) or 0) + float(m.group(3) or 0) / 1000
                return min(s, 120.0) + 0.5
    return min(2.0 ** attempt, 60.0)


class GroqAnswerer:
    """Same contract as ``ReactorAnswerer``, answered by a Groq-hosted LLM (OpenAI-compatible chat API).

    The message is exactly what Reactor would get (``reactor_service.build_message`` with a fresh ref), so
    the ``# FINAL ANSWER [ref:...]:`` line is parsed and cited the same way. Stateless: ``start_history``
    only returns a label. Retries 429 / 5xx / network errors with backoff, honouring retry-after.
    """

    def __init__(self, model: str | None = None, *, attempts: int = 6, timeout: float | None = None,
                 reasoning_effort: str | None = None, max_completion_tokens: int = 4096,
                 log: Callable[[str], None] = lambda s: None):
        self.model = model or GROQ_DEFAULT_MODEL
        self.attempts = max(1, attempts)
        self.timeout = timeout or 180.0
        self.reasoning_effort = reasoning_effort or os.environ.get("GROQ_REASONING_EFFORT", "medium")
        self.max_completion_tokens = max_completion_tokens
        self.log = log
        self.history_id: str | None = None
        self.last_reasoning = ""
        self.rate_limited = 0  # 429s seen, for reporting
        self._client = httpx.Client(base_url=GROQ_BASE_URL, timeout=self.timeout,
                                    headers={"Authorization": f"Bearer {groq_api_key()}"})

    def start_history(self, dataset_history_id: str) -> str:
        self.history_id = f"groq-{dataset_history_id}"
        return self.history_id

    def _payload(self, text: str) -> dict:
        p = {"model": self.model, "messages": [{"role": "user", "content": text}], "temperature": 0,
             "max_completion_tokens": self.max_completion_tokens}
        if self.model.startswith("openai/gpt-oss") and self.reasoning_effort:
            p["reasoning_effort"] = self.reasoning_effort
        return p

    def ask(self, r: Retrieved, composed: Composed) -> tuple[Answer, int]:
        from reactor_service.api import _evidence
        from reactor_service.protocol import citations, final_answers, is_abstention, new_ref

        ev = _evidence([e.model_dump() for e in composed.evidence])
        ref = new_ref()
        text = reactor.build_message(r.question, [e.model_dump() for e in ev],
                                     question_date=r.question_date or None, notes=composed.notes, ref=ref)
        retries = 0
        last = "no attempt"
        for attempt in range(1, self.attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.post("/chat/completions", json=self._payload(text))
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                self.log(f"{r.question_id}: network error on attempt {attempt}: {last}")
                retries += 1
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            latency = time.perf_counter() - t0
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _retry_after(resp, attempt)
                self.rate_limited += resp.status_code == 429
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self.log(f"{r.question_id}: HTTP {resp.status_code} on attempt {attempt}, waiting {wait:.1f}s")
                retries += 1
                time.sleep(wait)
                continue
            if resp.status_code != 200:  # 400/401/413...: a retry will not help
                raise NoAnswer(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
            msg = resp.json()["choices"][0]["message"]
            # gpt-oss writes citations as 【E1】; normalize so citation parsing / label stripping work as for Reactor
            content = (msg.get("content") or "").replace("\u3010", "[").replace("\u3011", "]")
            self.last_reasoning = msg.get("reasoning") or ""
            finals = final_answers(content)
            answer = finals.get(ref)
            if answer is None and finals:  # model mangled the ref: take its last final line
                answer = list(finals.values())[-1]
            if answer is None:
                last = "reply has no `# FINAL ANSWER` line"
                self.log(f"{r.question_id}: {last} on attempt {attempt}")
                retries += 1
                continue
            abstained = is_abstention(answer)
            cited, unknown = citations(answer, ev)
            a = Answer(answer=answer, abstained=abstained, citations=cited, unknown_citations=unknown,
                       uncited=bool(ev) and not cited and not abstained, reply=content, cycle=0,
                       latency_s=round(latency, 3), retries=retries, ref=ref, question=r.question,
                       question_date=r.question_date or None, question_id=r.question_id,
                       history_id=self.history_id)
            return a, attempt
        raise NoAnswer(f"Groq gave no answer after {self.attempts} attempts: {last}")


def make_answerer(provider: str, *, model: str | None = None, attempts: int | None = None,
                  timeout: float | None = None, log: Callable[[str], None] = lambda s: None) -> Answerer:
    if provider == "groq":
        return GroqAnswerer(model, attempts=attempts or 6, timeout=timeout, log=log)
    if provider == "reactor":
        return ReactorAnswerer(attempts=attempts or 3, timeout=timeout, log=log)
    raise ValueError(f"unknown provider {provider!r} (reactor | groq)")
