"""The loop: for each dataset history, a fresh Reactor history, then its questions one by one.

``run`` appends one ``QuestionResult`` line per question to ``results_path`` as soon as it is known, so an
interrupted run loses nothing and a rerun skips what is already there (delete lines to ask again).
``write_predictions`` turns a results file into ``predictions.jsonl``; it is a pure function of the results
and the ``Policy``, so policies can be changed without asking Reactor again.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from reactor_service import NoAnswer

from .answering import Answerer
from .data import append_jsonl, iter_jsonl, write_jsonl
from .models import Prediction, QuestionResult, Retrieved
from .postprocess import Policy, to_prediction
from .prompting import compose
from .retrieval import Retrieval

Log = Callable[[str], None]


def done_ids(results_path: str | Path) -> set[str]:
    p = Path(results_path)
    return {r["question_id"] for r in iter_jsonl(p)} if p.exists() else set()


def latest_results(results_path: str | Path) -> dict[str, QuestionResult]:
    """Load results without allowing a later transport failure to erase a valid answer."""
    results: dict[str, QuestionResult] = {}
    for row in iter_jsonl(results_path) if Path(results_path).exists() else []:
        candidate = QuestionResult.model_validate(row)
        current = results.get(candidate.question_id)
        if current is None or candidate.status != "no-answer" or current.status == "no-answer":
            results[candidate.question_id] = candidate
    return results


def reconcile_results(results_path: str | Path) -> int:
    """Compact duplicate rows, retaining a valid answer over a stale no-answer row."""
    path = Path(results_path)
    if not path.exists():
        return 0
    raw = list(iter_jsonl(path))
    resolved = latest_results(path)
    if len(raw) == len(resolved):
        return 0
    backup = path.with_suffix(path.suffix + ".before-reconcile")
    if not backup.exists():
        backup.write_text(path.read_text())
    write_jsonl(path, (result.model_dump() for result in resolved.values()))
    return len(raw) - len(resolved)


def reset_verification(results_path: str | Path) -> int:
    """Restore the most recent pre-verifier result for every question, preserving a backup."""
    path = Path(results_path)
    if not path.exists():
        return 0
    raw = list(iter_jsonl(path))
    restored: dict[str, QuestionResult] = {}
    for row in raw:
        candidate = QuestionResult.model_validate(row)
        if candidate.verification != "not-run":
            continue
        current = restored.get(candidate.question_id)
        if current is None or candidate.status != "no-answer" or current.status == "no-answer":
            restored[candidate.question_id] = candidate
    if not restored:
        return 0
    backup = path.with_suffix(path.suffix + ".before-reset-verification")
    if not backup.exists():
        backup.write_text(path.read_text())
    write_jsonl(path, (result.model_dump() for result in restored.values()))
    return len(raw) - len(restored)


def group_by_history(questions: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        groups[q["history_id"]].append(q)
    return dict(groups)


def answer_one(r: Retrieved, answerer: Answerer, max_chars: int, reactor_history_id: str) -> QuestionResult:
    c = compose(r, max_chars)
    base = dict(question_id=r.question_id, history_id=r.history_id, guessed_type=r.guessed_type,
                sent_ids=[e.id for e in c.evidence], candidates=r.candidates, message_chars=c.chars,
                reactor_history_id=reactor_history_id)
    try:
        a, attempts = answerer.ask(r, c)
    except NoAnswer as e:
        return QuestionResult(status="no-answer", error=f"{type(e).__name__}: {e}",
                              other_replies=e.other_replies, late_replies=e.late_replies,
                              attempts=getattr(answerer, "attempts", 1), **base)
    verify = getattr(answerer, "verify", None)
    verification = verify(r, c, a) if callable(verify) and not a.abstained else "not-run"
    unsupported = verification == "unsupported"
    return QuestionResult(
        status="abstained" if a.abstained or unsupported else "answered",
        answer_raw="INSUFFICIENT_EVIDENCE" if unsupported else a.answer,
        cited_ids=[] if unsupported else a.cited_ids,
        unknown_citations=a.unknown_citations, uncited=a.uncited, reply=a.reply,
        other_replies=a.other_replies, late_replies=a.late_replies, latency_s=a.latency_s,
        retries=a.retries, attempts=attempts, verification=verification, **base,
    )


def run(questions: list[dict], retrieval: Retrieval, answerer: Answerer, results_path: str | Path, *,
        max_chars: int = 16000, workers: int = 1, retry_ids: set[str] | None = None,
        log: Log = print) -> list[QuestionResult]:
    """Answer every question not yet in ``results_path``; returns the results produced by this call."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers > 1 and not getattr(answerer, "stateless", False):
        raise ValueError("parallel workers require a stateless answerer; Reactor must use --workers 1")
    skip = done_ids(results_path) - (retry_ids or set())
    todo = [q for q in questions if q["question_id"] not in skip]
    groups = group_by_history(todo)
    log(f"{len(todo)} question(s) in {len(groups)} dataset histories to ask ({len(skip)} already done)")
    if not todo:
        return []
    retrieved = {r.question_id: r for r in retrieval.retrieve(todo)}
    out: list[QuestionResult] = []
    if workers == 1:
        n = 0
        for hid, qs in groups.items():
            rh = answerer.start_history(hid)
            for q in qs:
                n += 1
                r = retrieved[q["question_id"]]
                res = answer_one(r, answerer, max_chars, rh)
                rh = getattr(answerer, "history_id", None) or rh  # the answerer may have had to restart it
                append_jsonl(results_path, res.model_dump())
                out.append(res)
                log(f"[{n}/{len(todo)}] {res.question_id} {res.status:9s} {res.latency_s:5.1f}s "
                    f"{res.answer_raw[:70]!r} cites={res.cited_ids}")
        return out

    def ask_stateless(q: dict) -> QuestionResult:
        r = retrieved[q["question_id"]]
        return answer_one(r, answerer, max_chars, answerer.start_history(r.history_id))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pipeline-answer") as executor:
        futures = [executor.submit(ask_stateless, q) for q in todo]
        for n, future in enumerate(as_completed(futures), 1):
            res = future.result()
            append_jsonl(results_path, res.model_dump())
            out.append(res)
            log(f"[{n}/{len(todo)}] {res.question_id} {res.status:9s} {res.latency_s:5.1f}s "
                f"{res.answer_raw[:70]!r} cites={res.cited_ids}")
    return out


def verify_existing(questions: list[dict], retrieval: Retrieval, answerer: Answerer, results_path: str | Path, *,
                    max_chars: int = 16000, workers: int = 1, log: Log = print) -> list[QuestionResult]:
    """Verify saved answered rows without repeating the original answer request."""
    verify = getattr(answerer, "verify_candidate", None)
    if not callable(verify):
        raise ValueError("the selected answerer does not support answer verification")
    if workers < 1 or (workers > 1 and not getattr(answerer, "stateless", False)):
        raise ValueError("parallel verification requires a stateless OpenAI or Groq answerer")
    current = latest_results(results_path)
    todo = [q for q in questions if q["question_id"] in current and current[q["question_id"]].status == "answered"
            and current[q["question_id"]].verification not in ("supported", "unsupported")]
    log(f"verifying {len(todo)} saved answer(s)")
    if not todo:
        return []
    retrieved = {row.question_id: row for row in retrieval.retrieve(todo)}

    def one(q: dict) -> QuestionResult:
        previous = current[q["question_id"]]
        composed = compose(retrieved[q["question_id"]], max_chars)
        verdict = verify(retrieved[q["question_id"]], composed, previous.answer_raw, previous.cited_ids)
        if verdict == "unsupported":
            return previous.model_copy(update={
                "status": "abstained", "answer_raw": "INSUFFICIENT_EVIDENCE", "cited_ids": [],
                "uncited": False, "verification": verdict,
            })
        return previous.model_copy(update={"verification": verdict})

    out: list[QuestionResult] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pipeline-verify") as executor:
        futures = [executor.submit(one, q) for q in todo]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            append_jsonl(results_path, result.model_dump())
            out.append(result)
            log(f"[{n}/{len(todo)}] {result.question_id} verification={result.verification}")
    return out


def write_predictions(results_path: str | Path, out_path: str | Path, *, questions: list[dict],
                      policy: Policy = Policy(), sessions: dict[str, dict[str, str]] | None = None,
                      fill_missing: bool = True, log: Log = print) -> list[Prediction]:
    """results.jsonl -> predictions.jsonl with exactly one line per question, in questions.jsonl order.

    A later result for the same question_id replaces an earlier one (so appending a retry is enough).
    ``sessions`` (``{history_id: {session_id: date}}``) restricts citations to the question's history.
    Questions without a result are written as abstentions when ``fill_missing`` (a missing line scores as
    wrong anyway), and are listed in the log.
    """
    results = latest_results(results_path)
    preds: list[Prediction] = []
    missing: list[str] = []
    for q in questions:
        qid = q["question_id"]
        r = results.get(qid)
        if r is None:
            missing.append(qid)
            if fill_missing:
                preds.append(Prediction(question_id=qid, abstain=True, answer="", citations=[]))
            continue
        known = set(sessions[q["history_id"]]) if sessions and q["history_id"] in sessions else None
        preds.append(to_prediction(r, policy, known))
    write_jsonl(out_path, (p.model_dump() for p in preds))
    abst = sum(p.abstain for p in preds)
    log(f"{len(preds)} prediction(s) ({abst} abstained) -> {out_path}"
        + (f"; {len(missing)} question(s) had no result: {missing[:5]}{'...' if len(missing) > 5 else ''}"
           if missing else ""))
    return preds
