"""``python -m memtest`` entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from memtest.data import DataError
from memtest.kit import SCORE_PY
from memtest.pack import MAX_CHARS


def _add_retrieval_args(parser: argparse.ArgumentParser, *, online: bool) -> None:
    parser.add_argument("--split", default="dev", help="data/<split>/ directory (default: dev)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="directory that contains the split")
    parser.add_argument("--retriever", default="bm25", help="retriever name (default: bm25)")
    parser.add_argument("--limit", type=int, default=None, help="first N questions in file order")
    parser.add_argument("--top-sessions", type=int, default=12, help="sessions the retriever may return")
    parser.add_argument("--turns-per-session", type=int, default=4, help="highest-scoring turns kept per session")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS, help="client message budget (default: 16000)")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where timestamped run directories go")
    if not online:
        parser.add_argument(
            "--abstain-below",
            type=float,
            default=0.0,
            help="abstain when the top snippet score is at or below this value (default: 0)",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memtest",
        description="Retrieval evaluation for the RxAI memory challenge. "
        "Offline metrics do not call Reactor. Online answers do, through vendor/rxai/client.py.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser("offline", help="BM25 (or another retriever) against dev gold")
    _add_retrieval_args(offline, online=False)

    online = sub.add_parser("online", help="ask Reactor and write predictions.jsonl")
    _add_retrieval_args(online, online=True)
    online.add_argument("--results", type=Path, default=None, help="results.jsonl to resume; default is a new run")
    online.add_argument("--base-url", default=None, help="or set REACTOR_BASE_URL")
    online.add_argument("--agent-id", default=None, help="or set REACTOR_AGENT_ID")
    online.add_argument(
        "--abstain-on",
        default="agent,no-answer",
        help="passed to predictions.py (default: agent,no-answer)",
    )

    score = sub.add_parser("score", help="run the official score.py on a dev predictions file")
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--details", type=Path, default=None, help="optional per-question JSONL from score.py")
    return parser


def run_official_score(gold: Path, predictions: Path, details: Path | None) -> int:
    """Run vendored ``score.py`` and let its table through to this process."""

    if not SCORE_PY.is_file():
        print(f"vendored score.py is missing: {SCORE_PY}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(SCORE_PY), str(gold), str(predictions)]
    if details is not None:
        cmd += ["--details", str(details)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            return run_official_score(args.gold, args.predictions, args.details)
        if args.command == "offline":
            from memtest.offline import run_offline

            code, _directory = run_offline(
                data_dir=args.data_dir,
                split=args.split,
                retriever_name=args.retriever,
                limit=args.limit,
                top_sessions=args.top_sessions,
                turns_per_session=args.turns_per_session,
                max_chars=args.max_chars,
                abstain_below=args.abstain_below,
                runs_dir=args.runs_dir,
            )
            return code
        if args.command == "online":
            from memtest.online import run_online

            return run_online(
                data_dir=args.data_dir,
                split=args.split,
                retriever_name=args.retriever,
                limit=args.limit,
                top_sessions=args.top_sessions,
                turns_per_session=args.turns_per_session,
                max_chars=args.max_chars,
                runs_dir=args.runs_dir,
                results_path=args.results,
                base_url=args.base_url,
                agent_id=args.agent_id,
                abstain_on=args.abstain_on,
            )
    except (DataError, KeyError, ImportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command}")
    return 2
