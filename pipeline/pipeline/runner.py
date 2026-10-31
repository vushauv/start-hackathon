"""The loop: for each dataset history, a fresh Reactor history, then its questions one by one.

``run`` appends one ``QuestionResult`` line per question to ``results_path`` as soon as it is known, so an
interrupted run loses nothing and a rerun skips what is already there (delete lines to ask again).
``write_predictions`` turns a results file into ``predictions.jsonl``; it is a pure function of the results
and the ``Policy``, so policies can be changed without asking Reactor again.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
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
    return QuestionResult(
        status="abstained" if a.abstained else "answered", answer_raw=a.answer, cited_ids=a.cited_ids,
        unknown_citations=a.unknown_citations, uncited=a.uncited, reply=a.reply,
        other_replies=a.other_replies, late_replies=a.late_replies, latency_s=a.latency_s,
        retries=a.retries, attempts=attempts, **base,
    )


def run(questions: list[dict], retrieval: Retrieval, answerer: Answerer, results_path: str | Path, *,
        max_chars: int = 16000, log: Log = print) -> list[QuestionResult]:
    """Answer every question not yet in ``results_path``; returns the results produced by this call."""
    skip = done_ids(results_path)
    todo = [q for q in questions if q["question_id"] not in skip]
    groups = group_by_history(todo)
    log(f"{len(todo)} question(s) in {len(groups)} dataset histories to ask ({len(skip)} already done)")
    if not todo:
        return []
    retrieved = {r.question_id: r for r in retrieval.retrieve(todo)}
    out: list[QuestionResult] = []
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


def write_predictions(results_path: str | Path, out_path: str | Path, *, questions: list[dict],
                      policy: Policy = Policy(), sessions: dict[str, dict[str, str]] | None = None,
                      fill_missing: bool = True, log: Log = print) -> list[Prediction]:
    """results.jsonl -> predictions.jsonl with exactly one line per question, in questions.jsonl order.

    A later result for the same question_id replaces an earlier one (so appending a retry is enough).
    ``sessions`` (``{history_id: {session_id: date}}``) restricts citations to the question's history.
    Questions without a result are written as abstentions when ``fill_missing`` (a missing line scores as
    wrong anyway), and are listed in the log.
    """
    results: dict[str, QuestionResult] = {}
    for row in iter_jsonl(results_path) if Path(results_path).exists() else []:
        r = QuestionResult.model_validate(row)
        results[r.question_id] = r
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
