"""Ask Reactor through the vendored client, then write predictions.jsonl.

Credentials are never read from the command line. ``client.py`` itself reads
``REACTOR_ACCESS_KEY``. The base URL and agent id come from flags or from
``REACTOR_BASE_URL`` and ``REACTOR_AGENT_ID``. If any of the three is missing
the run stops before it opens a socket and exits 0.

Results are appended to ``results.jsonl`` after every question. A later run
with the same file skips question ids already in it. A new Reactor history is
started for each dataset history that still has a pending question, which
matches the participant kit: a resumed history does not keep the agent's
memory from the previous process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from memtest.data import DataError, HistoryStore, iter_jsonl, load_questions, resolve_split, select_questions
from memtest.kit import CLIENT_PY, PREDICTIONS_PY
from memtest.offline import run_dir
from memtest.pack import client_passages, pack_evidence
from memtest.retriever import build_retriever


def missing_credentials(base_url: str | None, agent_id: str | None) -> list[str]:
    """Names of the Reactor settings that are still empty. The key's value is not returned."""

    missing = []
    if not os.environ.get("REACTOR_ACCESS_KEY", "").strip():
        missing.append("REACTOR_ACCESS_KEY")
    if not (base_url or os.environ.get("REACTOR_BASE_URL", "")).strip():
        missing.append("REACTOR_BASE_URL")
    if not (agent_id or os.environ.get("REACTOR_AGENT_ID", "")).strip():
        missing.append("REACTOR_AGENT_ID")
    return missing


def resolve_endpoint(base_url: str | None, agent_id: str | None) -> tuple[str, str]:
    base = (base_url or os.environ.get("REACTOR_BASE_URL", "")).strip()
    agent = (agent_id or os.environ.get("REACTOR_AGENT_ID", "")).strip()
    return base, agent


def load_done(path: Path) -> set[str]:
    """Question ids already recorded. Two lines for one id is an error: predictions.py would refuse the file."""

    if not path.exists():
        return set()
    seen: set[str] = set()
    for record in iter_jsonl(path):
        qid = record.get("question_id")
        if not isinstance(qid, str) or not qid:
            continue
        if qid in seen:
            raise DataError(
                f"{path} has two results for {qid}. Keep one line (predictions.py rejects duplicates), then rerun."
            )
        seen.add(qid)
    return seen


def default_invoke(base_url: str, agent_id: str, extra: list[str]) -> dict:
    """Run vendored ``client.py``. The access key stays in the environment."""

    proc = subprocess.run(
        [sys.executable, str(CLIENT_PY), "--base-url", base_url, "--agent-id", agent_id, *extra],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"status": "error", "error": (proc.stderr or proc.stdout or "client.py produced no JSON").strip()[-500:]}
    if not isinstance(result, dict):
        return {"status": "error", "error": "client.py stdout was not a JSON object"}
    return result


def write_predictions(results: Path, histories: Path, out: Path, abstain_on: str) -> int:
    """Delegate to vendored ``predictions.py``. Exit 1 (skipped rows) is still a written file."""

    cmd = [
        sys.executable,
        str(PREDICTIONS_PY),
        str(results),
        "--histories",
        str(histories),
        "--out",
        str(out),
    ]
    if abstain_on.strip():
        cmd += ["--abstain-on", abstain_on.strip()]
    proc = subprocess.run(cmd)
    if proc.returncode == 1:
        print("predictions.py skipped some results; the file contains the rest", file=sys.stderr)
        return 0
    return proc.returncode


def run_online(
    *,
    data_dir: Path,
    split: str,
    retriever_name: str,
    limit: int | None,
    top_sessions: int,
    turns_per_session: int,
    max_chars: int,
    runs_dir: Path,
    results_path: Path | None,
    base_url: str | None,
    agent_id: str | None,
    abstain_on: str,
    invoke=None,
) -> int:
    missing = missing_credentials(base_url, agent_id)
    if missing:
        print(
            "Reactor credentials are not configured (missing "
            + ", ".join(missing)
            + "). Skipping the online run. Set them in the environment and rerun; nothing was called.",
            file=sys.stderr,
        )
        return 0

    endpoint_url, endpoint_agent = resolve_endpoint(base_url, agent_id)
    paths = resolve_split(data_dir, split, need_gold=False)
    questions = select_questions(load_questions(paths.questions), limit)
    if results_path is None:
        directory = run_dir(runs_dir)
        results_path = directory / "results.jsonl"
        predictions_path = directory / "predictions.jsonl"
        meta = directory / "run.json"
    else:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path = results_path.with_name("predictions.jsonl")
        meta = results_path.with_name("run.json")

    done = load_done(results_path)
    pending = [q for q in questions if q["question_id"] not in done]
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for question in pending:
        groups.setdefault(question["history_id"], []).append(question)
    print(
        f"{len(pending)} question(s) to ask in {len(groups)} dataset histories "
        f"({len(done)} already in {results_path})",
        file=sys.stderr,
    )

    caller = invoke or (lambda extra: default_invoke(endpoint_url, endpoint_agent, extra))
    retriever = build_retriever(retriever_name, top_sessions=top_sessions, turns_per_session=turns_per_session)
    store = HistoryStore(paths.histories) if pending else None

    meta.write_text(json.dumps({
        "command": "online",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retriever": retriever_name,
        "split": split,
        "data_dir": str(paths.root),
        "limit": limit,
        "max_chars": max_chars,
        "top_sessions": top_sessions,
        "turns_per_session": turns_per_session,
        "base_url": endpoint_url,
        "agent_id": endpoint_agent,
        "results": str(results_path),
    }, indent=2) + "\n", encoding="utf-8")

    status = 0
    asked = False
    if pending and store is not None:
        with tempfile.TemporaryDirectory() as tmp, results_path.open("a", encoding="utf-8") as out:
            evidence_n = 0
            for history_id, group in groups.items():
                opened = caller(["--new-history"])
                if opened.get("status") != "history" or not opened.get("history_id"):
                    detail = opened.get("error") or opened
                    print(f"--new-history failed for {history_id}: {detail}", file=sys.stderr)
                    status = 2
                    break
                reactor_history = opened["history_id"]
                try:
                    history = store.get(history_id)
                except DataError as exc:
                    print(str(exc), file=sys.stderr)
                    status = 2
                    break
                for question in group:
                    snippets = retriever.retrieve(question, history)
                    packed = pack_evidence(
                        question["question"], question["question_date"], snippets, max_chars=max_chars
                    )
                    passages = client_passages(packed)
                    extra = [
                        "--history", reactor_history,
                        "--question", question["question"],
                        "--question-date", question["question_date"],
                        "--question-id", question["question_id"],
                        "--max-chars", str(max_chars),
                    ]
                    if passages:
                        evidence_n += 1
                        evidence_path = Path(tmp) / f"evidence-{evidence_n}.json"
                        evidence_path.write_text(json.dumps(passages, ensure_ascii=False), encoding="utf-8")
                        extra += ["--evidence", str(evidence_path)]
                    result = caller(extra)
                    if not isinstance(result, dict):
                        result = {"status": "error", "error": "invoke did not return an object"}
                    result.setdefault("question_id", question["question_id"])
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    asked = True
                    print(
                        f"{question['question_id']}: {result.get('status')} "
                        f"{str(result.get('answer', ''))[:60]}",
                        file=sys.stderr,
                    )

    if results_path.exists() and results_path.stat().st_size:
        code = write_predictions(results_path, paths.histories, predictions_path, abstain_on)
        if status == 0:
            status = code
        if code == 0:
            print(f"wrote {predictions_path}")
    elif status == 0 and not asked:
        print("no results to convert", file=sys.stderr)
    return status
