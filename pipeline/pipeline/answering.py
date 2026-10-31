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

    def verify(self, r: Retrieved, composed: Composed, answer: Answer) -> str:
        """Return supported, unsupported, or unverified for a completed answer."""
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
                    # A closed socket can leave the server-side history in an
                    # unusable state. Start clean before the next attempt rather
                    # than sending another request to the closed history.
                    if attempt < self.attempts:
                        self.start_history(r.history_id)
                    time.sleep(3)
        assert last is not None
        raise last


class DryRunAnswerer:
    """No network: answers with the first passage cited, so the plumbing can be exercised end to end."""

    stateless = True

    def start_history(self, dataset_history_id: str) -> str:
        return f"dry-{dataset_history_id}"

    def ask(self, r: Retrieved, composed: Composed) -> tuple[Answer, int]:
        from reactor_service import Citation

        cited = [Citation(label="E1", id=composed.evidence[0].id)] if composed.evidence else []
        a = Answer(answer="(dry run) [E1]" if cited else "(dry run)", abstained=False, citations=cited,
                   reply="", cycle=0, latency_s=0.0, ref="00000000", question=r.question,
                   question_date=r.question_date or None, question_id=r.question_id,
                   history_id=f"dry-{r.history_id}", uncited=False)
        return a, 1


# ---------------------------------------------------------------------------------------------- Groq
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Picked from GET /v1/models (Oct 2026): the strongest listed model, 131072-token context window.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_DEFAULT_MAX_CHARS = 60000
GROQ_ENV_FILE = Path(__file__).resolve().parent.parent / ".env.groq"


def _env_key(name: str, env_file: Path) -> str:
    key = os.environ.get(name, "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(name + "="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise reactor.ConfigError(f"{name} not set (environment or {env_file})")
    return key


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
                 verify_answers: bool = False,
                 log: Callable[[str], None] = lambda s: None):
        self.model = model or GROQ_DEFAULT_MODEL
        self.attempts = max(1, attempts)
        self.timeout = timeout or 180.0
        self.reasoning_effort = reasoning_effort or os.environ.get("GROQ_REASONING_EFFORT", "medium")
        self.max_completion_tokens = max_completion_tokens
        self.verify_answers = verify_answers
        self.log = log
        self.history_id: str | None = None
        self.last_reasoning = ""
        self.rate_limited = 0  # 429s seen, for reporting
        self._client = httpx.Client(base_url=self.BASE_URL, timeout=self.timeout,
                                    headers={"Authorization": f"Bearer {self._key()}"})

    BASE_URL = GROQ_BASE_URL
    PREFIX = "groq"
    stateless = True

    def _key(self) -> str:
        return groq_api_key()

    def start_history(self, dataset_history_id: str) -> str:
        return f"{self.PREFIX}-{dataset_history_id}"

    def _payload(self, text: str, *, reasoning_effort: str | None = None) -> dict:
        p = {"model": self.model, "messages": [{"role": "user", "content": text}], "temperature": 0,
             "max_completion_tokens": self.max_completion_tokens}
        if self.model.startswith("openai/gpt-oss") and self.reasoning_effort:
            p["reasoning_effort"] = reasoning_effort or self.reasoning_effort
        return p

    def ask(self, r: Retrieved, composed: Composed) -> tuple[Answer, int]:
        from reactor_service.api import _evidence
        from reactor_service.protocol import final_answers, final_output, is_abstention, new_ref

        ev = _evidence([e.model_dump() for e in composed.evidence])
        ref = new_ref()
        text = reactor.build_message(r.question, [e.model_dump() for e in ev],
                                     question_date=r.question_date or None, notes=composed.notes, ref=ref)
        retries = 0
        last = "no attempt"
        for attempt in range(1, self.attempts + 1):
            t0 = time.perf_counter()
            try:
                # Spend extra deliberation only on questions that require combining
                # dated or multiple sessions; direct fact lookups stay fast and cheap.
                effort = self.reasoning_effort
                if r.guessed_type in ("temporal-reasoning", "multi-session"):
                    effort = os.environ.get("OPENAI_REASONING_EFFORT_HARD", "medium")
                resp = self._client.post("/chat/completions", json=self._payload(text, reasoning_effort=effort))
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
                raise NoAnswer(f"{self.PREFIX} HTTP {resp.status_code}: {resp.text[:300]}")
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
            answer, cited, unknown = final_output(answer, ev)
            abstained = is_abstention(answer)
            a = Answer(answer=answer, abstained=abstained, citations=cited, unknown_citations=unknown,
                       uncited=bool(ev) and not cited and not abstained, reply=content, cycle=0,
                       latency_s=round(latency, 3), retries=retries, ref=ref, question=r.question,
                       question_date=r.question_date or None, question_id=r.question_id,
                       history_id=f"{self.PREFIX}-{r.history_id}")
            return a, attempt
        raise NoAnswer(f"Groq gave no answer after {self.attempts} attempts: {last}")

    def verify(self, r: Retrieved, composed: Composed, answer: Answer) -> str:
        """Use an independent, citation-focused pass to reject unsupported final answers."""
        return self.verify_candidate(r, composed, answer.answer, answer.cited_ids)

    def verify_candidate(self, r: Retrieved, composed: Composed, answer_text: str, cited_ids: list[str]) -> str:
        """Check a saved answer against its cited evidence without asking the original question again."""
        if not self.verify_answers:
            return "not-run"
        cited = set(cited_ids)
        passages = [e for e in composed.evidence if not cited or e.id in cited]
        evidence = "\n\n".join(
            f"[E{i}] date: {e.date}\n{e.text}" for i, e in enumerate(passages, 1)
        )
        prompt = (
            "Check whether the candidate answer is fully supported by the supplied evidence. "
            "Return exactly SUPPORTED when every required fact follows from the evidence. "
            "Return exactly UNSUPPORTED when any required fact is absent, contradicted, or only guessed.\n\n"
            f"Question: {r.question}\nCandidate answer: {answer_text}\n\nEvidence:\n{evidence}"
        )
        try:
            response = self._client.post("/chat/completions", json=self._payload(prompt))
            if response.status_code != 200:
                self.log(f"{r.question_id}: verifier HTTP {response.status_code}; keeping answer")
                return "unverified"
            content = response.json()["choices"][0]["message"].get("content") or ""
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            self.log(f"{r.question_id}: verifier {type(exc).__name__}; keeping answer")
            return "unverified"
        verdict = content.strip().upper()
        if verdict == "SUPPORTED":
            return "supported"
        if verdict == "UNSUPPORTED":
            return "unsupported"
        self.log(f"{r.question_id}: verifier returned {content[:80]!r}; keeping answer")
        return "unverified"


OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-5.6-luna"
OPENAI_ENV_FILE = Path(__file__).resolve().parent.parent / ".env.openai"


class OpenAIAnswerer(GroqAnswerer):
    """Same as GroqAnswerer against api.openai.com (key: OPENAI_API_KEY or pipeline/.env.openai)."""
    BASE_URL = OPENAI_BASE_URL
    PREFIX = "openai"

    def __init__(self, model: str | None = None, **kw):
        super().__init__(model or OPENAI_DEFAULT_MODEL, **kw)
        self.reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "low")

    def _key(self) -> str:
        return _env_key("OPENAI_API_KEY", OPENAI_ENV_FILE)

    def _payload(self, text: str, *, reasoning_effort: str | None = None) -> dict:
        p = {"model": self.model, "messages": [{"role": "user", "content": text}],
             "max_completion_tokens": self.max_completion_tokens}
        if self.model.startswith(("gpt-5", "gpt-6", "o")):
            p["reasoning_effort"] = reasoning_effort or self.reasoning_effort
        else:
            p["temperature"] = 0
        return p


def make_answerer(provider: str, *, model: str | None = None, attempts: int | None = None,
                  timeout: float | None = None, verify_answers: bool = False,
                  log: Callable[[str], None] = lambda s: None) -> Answerer:
    if provider == "groq":
        return GroqAnswerer(model, attempts=attempts or 6, timeout=timeout, verify_answers=verify_answers, log=log)
    if provider == "openai":
        return OpenAIAnswerer(model, attempts=attempts or 6, timeout=timeout, verify_answers=verify_answers, log=log)
    if provider == "reactor":
        return ReactorAnswerer(attempts=attempts or 3, timeout=timeout, log=log)
    raise ValueError(f"unknown provider {provider!r} (reactor | groq)")
