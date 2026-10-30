"""Score a retriever against dev gold without calling Reactor."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from memtest.data import DataError, HistoryStore, load_gold, load_questions, resolve_split, select_questions
from memtest.metrics import DEFAULT_K, decides_abstain, format_table, score_question, summarize
from memtest.pack import client_passages, message_length, pack_evidence
from memtest.retriever import build_retriever, unique_session_ids


def run_dir(runs_dir: Path | str) -> Path:
    """``runs/<UTC timestamp>/``, with a numeric suffix if that second is taken."""

    root = Path(runs_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / stamp
    suffix = 2
    while path.exists():
        path = root / f"{stamp}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def _git_rev() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    rev = proc.stdout.strip()
    return rev or None


def evaluate(
    questions: list[dict],
    gold_by_id: dict[str, dict],
    store: HistoryStore,
    retriever,
    *,
    max_chars: int,
    abstain_below: float,
    k_values: tuple[int, ...] = DEFAULT_K,
) -> list[dict]:
    """One metrics record per question that has a gold row."""

    records = []
    missing = 0
    for index, question in enumerate(questions, 1):
        gold = gold_by_id.get(question["question_id"])
        if gold is None:
            missing += 1
            continue
        history = store.get(question["history_id"])
        snippets = retriever.retrieve(question, history)
        packed = pack_evidence(question["question"], question["question_date"], snippets, max_chars=max_chars)
        passages = client_passages(packed)
        try:
            chars = message_length(question["question"], question["question_date"], passages)
        except ValueError:
            chars = None
        top = snippets[0]["score"] if snippets else None
        records.append(score_question(
            question_id=question["question_id"],
            question_type=gold["question_type"],
            gold_abstain=bool(gold["abstain"]),
            gold_evidence=list(gold.get("evidence") or []),
            pred_abstain=decides_abstain(snippets, abstain_below),
            ranked=unique_session_ids(snippets),
            sent=unique_session_ids(packed),
            top_score=None if top is None else float(top),
            message_chars=chars,
            k_values=k_values,
        ))
        if index == 1 or index % 25 == 0 or index == len(questions):
            print(f"scored {index}/{len(questions)}", file=sys.stderr, flush=True)
    if missing:
        print(f"{missing} question(s) had no gold row and were skipped", file=sys.stderr)
    if not records:
        raise DataError("no questions to score (gold did not match this selection)")
    return records


def write_report(directory: Path, report: dict, records: list[dict]) -> None:
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (directory / "details.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_offline(
    *,
    data_dir: Path,
    split: str,
    retriever_name: str,
    limit: int | None,
    top_sessions: int,
    turns_per_session: int,
    max_chars: int,
    abstain_below: float,
    runs_dir: Path,
    k_values: tuple[int, ...] = DEFAULT_K,
) -> tuple[int, Path]:
    paths = resolve_split(data_dir, split, need_gold=True)
    questions = select_questions(load_questions(paths.questions), limit)
    gold = load_gold(paths.gold)
    print(f"indexing {paths.histories}", file=sys.stderr, flush=True)
    store = HistoryStore(paths.histories)
    retriever = build_retriever(retriever_name, top_sessions=top_sessions, turns_per_session=turns_per_session)
    records = evaluate(
        questions,
        gold,
        store,
        retriever,
        max_chars=max_chars,
        abstain_below=abstain_below,
        k_values=k_values,
    )
    rows = summarize(records, k_values)
    directory = run_dir(runs_dir)
    report = {
        "command": "offline",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_rev": _git_rev(),
        "retriever": retriever_name,
        "split": split,
        "data_dir": str(paths.root),
        "limit": limit,
        "n_questions": len(records),
        "top_sessions": top_sessions,
        "turns_per_session": turns_per_session,
        "max_chars": max_chars,
        "abstain_below": abstain_below,
        "k": list(k_values),
        "metrics": rows,
    }
    write_report(directory, report, records)
    print(f"wrote {directory / 'report.json'}")
    print(format_table(rows, k_values))
    return 0, directory
