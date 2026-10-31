"""``pipeline`` command line: retrieve -> answer -> predict -> score, each also runnable on its own.

    pipeline retrieve --split test                 # GPU box: writes runs/test/evidence.jsonl
    pipeline answer   --split test                 # any box: asks Reactor, appends runs/test/results.jsonl
    pipeline predict  --split test                 # results.jsonl -> runs/test/predictions.jsonl
    pipeline score    --split dev                  # dev only: the organizers' metrics
    pipeline run      --split dev --limit 20       # all of the above that apply
    pipeline run      --split dev --provider groq  # same, answered by a Groq LLM -> runs/dev/groq/<model>/

Files live in ``pipeline/runs/<split>/`` unless overridden with --evidence/--results/--predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .answering import GROQ_DEFAULT_MAX_CHARS, GROQ_DEFAULT_MODEL, make_answerer
from .data import Split, split_dir
from .postprocess import Policy
from .prompting import compose
from .retrieval import FileRetrieval, LocalRetrieval, save_evidence
from .runner import run, write_predictions

RUNS = Path(__file__).resolve().parent.parent / "runs"


def _provider_defaults(a: argparse.Namespace) -> None:
    """Fill --model / --max-chars from the provider when not given."""
    groq = getattr(a, "provider", "reactor") == "groq"
    if hasattr(a, "model") and not a.model and groq:
        a.model = GROQ_DEFAULT_MODEL
    if hasattr(a, "max_chars") and a.max_chars is None:
        a.max_chars = GROQ_DEFAULT_MAX_CHARS if groq else 16000


def _out_dir(a: argparse.Namespace) -> Path:
    """runs/<split>/ for Reactor; runs/<split>/groq/<model>/ for Groq, so the two never mix."""
    d = RUNS / a.split
    if getattr(a, "provider", "reactor") == "groq":
        d = d / "groq" / (getattr(a, "model", None) or GROQ_DEFAULT_MODEL).replace("/", "_")
    return d


def _paths(a: argparse.Namespace) -> dict[str, Path]:
    _provider_defaults(a)
    d = _out_dir(a)
    ev = RUNS / a.split / "evidence.jsonl"  # evidence is shared by all providers
    return {
        "evidence": Path(a.evidence) if getattr(a, "evidence", None) else ev,
        "results": Path(a.results) if getattr(a, "results", None) else d / "results.jsonl",
        "predictions": Path(a.predictions) if getattr(a, "predictions", None) else d / "predictions.jsonl",
        "details": d / "score_details.jsonl",
    }


def _questions(split: Split, a: argparse.Namespace) -> list[dict]:
    qs = split.questions
    ids = set(getattr(a, "question_ids", None) or [])
    if ids:
        qs = [q for q in qs if q["question_id"] in ids]
    hids = set(getattr(a, "history_ids", None) or [])
    if hids:
        qs = [q for q in qs if q["history_id"] in hids]
    limit = getattr(a, "limit", None)
    return qs[:limit] if limit else qs


def _log(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def cmd_retrieve(a: argparse.Namespace) -> int:
    d = split_dir(a.split, a.data)
    split = Split(d)
    ret = LocalRetrieval(d, a.cache, reranker=None if a.reranker == "none" else a.reranker,
                         rerank_text=a.rerank_text, fusion=a.fusion, device=a.device, log=_log)
    rows = ret.retrieve(_questions(split, a))
    out = _paths(a)["evidence"]
    save_evidence(rows, out)
    chars = [sum(len(e.text) for e in r.evidence) for r in rows] or [0]
    _log(f"wrote {len(rows)} rows -> {out}; evidence chars max={max(chars)} mean={sum(chars) / len(chars):.0f}")
    return 0


def cmd_answer(a: argparse.Namespace) -> int:
    split = Split(split_dir(a.split, a.data))
    p = _paths(a)
    qs = _questions(split, a)
    p["results"].parent.mkdir(parents=True, exist_ok=True)
    retrieval = FileRetrieval(p["evidence"])
    if a.dry_run:
        rows = retrieval.retrieve(qs)
        sizes = [compose(r, a.max_chars).chars for r in rows]
        _log(f"dry run: {len(rows)} messages, chars max={max(sizes)} mean={sum(sizes) / len(sizes):.0f} "
             f"(limit {a.max_chars}); nothing sent")
        if a.show:
            print(compose(rows[0], a.max_chars).message)
        return 0
    answerer = make_answerer(a.provider, model=a.model, attempts=a.attempts, timeout=a.timeout, log=_log)
    _log(f"provider {a.provider}" + (f", model {a.model}" if a.provider == "groq" else "")
         + f", max-chars {a.max_chars} -> {p['results']}")
    run(qs, retrieval, answerer, p["results"], max_chars=a.max_chars, log=_log)
    return 0


def cmd_predict(a: argparse.Namespace) -> int:
    split = Split(split_dir(a.split, a.data))
    p = _paths(a)
    policy = Policy(fallback_citations=a.fallback_citations, abstain_citations=a.abstain_citations,
                    abstain_on_no_answer=not a.empty_on_no_answer)
    write_predictions(p["results"], p["predictions"], questions=_questions(split, a), policy=policy,
                      sessions=split.sessions() if a.check_sessions else None, log=_log)
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    from .scoring import format_table, score

    d = split_dir(a.split, a.data)
    gold = d / "gold.jsonl"
    if not gold.exists():
        _log(f"no gold.jsonl in {d}: scoring is only possible on dev splits")
        return 2
    p = _paths(a)
    print(format_table(score(gold, p["predictions"], p["details"])))
    _log(f"per-question details -> {p['details']}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    p = _paths(a)
    if not p["evidence"].exists():
        _log(f"{p['evidence']} missing: running retrieval (needs the `retrieval` extra and a GPU)")
        cmd_retrieve(a)
    cmd_answer(a)
    if a.dry_run:
        return 0
    cmd_predict(a)
    gold = split_dir(a.split, a.data) / "gold.jsonl"
    return cmd_score(a) if gold.exists() else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def provider_args(sp):
        sp.add_argument("--provider", choices=["reactor", "groq"], default="reactor",
                        help="who answers (default reactor). groq: files go to runs/<split>/groq/<model>/")
        sp.add_argument("--model", help=f"groq model (default {GROQ_DEFAULT_MODEL})")

    def common(sp, *, selection=True):
        sp.add_argument("--split", required=True, help="dev, dev-large, test, final ...")
        sp.add_argument("--data", help="directory with the split's files (default: $PIPELINE_DATA_ROOT/<split>)")
        sp.add_argument("--evidence", help="evidence JSONL (default runs/<split>/evidence.jsonl)")
        sp.add_argument("--results", help="results JSONL (default runs/<split>/results.jsonl)")
        sp.add_argument("--predictions", help="predictions JSONL (default runs/<split>/predictions.jsonl)")
        provider_args(sp)
        if selection:
            sp.add_argument("--limit", type=int, help="only the first N questions")
            sp.add_argument("--question-ids", nargs="*", help="only these question ids")
            sp.add_argument("--history-ids", nargs="*", help="only questions of these dataset histories")

    def retrieve_args(sp):
        sp.add_argument("--cache", help="embedding cache dir (default retriever/cache/<split>)")
        sp.add_argument("--reranker", default="base", help="base | v2-m3 | none")
        sp.add_argument("--rerank-text", default="best2", choices=["best1", "best2"])
        sp.add_argument("--fusion", default="fused", choices=["fused", "rerank"])
        sp.add_argument("--device", default="cuda")

    def answer_args(sp):
        sp.add_argument("--max-chars", type=int, default=None,
                        help=f"message limit (default 16000 for reactor, server allows 32000; {GROQ_DEFAULT_MAX_CHARS} for groq)")
        sp.add_argument("--attempts", type=int, help="tries per question before giving up (default 3 reactor, 6 groq)")
        sp.add_argument("--timeout", type=float, help="seconds to wait per try (default REACTOR_TIMEOUT; 180 for groq)")
        sp.add_argument("--dry-run", action="store_true", help="compose messages, report sizes, send nothing")
        sp.add_argument("--show", action="store_true", help="with --dry-run: print the first message")

    def predict_args(sp):
        sp.add_argument("--fallback-citations", type=int, default=1, help="cite top-N sent passages when the answer cites none")
        sp.add_argument("--abstain-citations", type=int, default=1, help="cite top-N sent passages on abstention")
        sp.add_argument("--empty-on-no-answer", action="store_true", help="submit '' instead of abstaining when Reactor gave no answer")
        sp.add_argument("--no-check-sessions", dest="check_sessions", action="store_false",
                        help="skip restricting citations to the history's session ids (avoids reading histories.jsonl)")

    sp = sub.add_parser("retrieve", help="build evidence with the GPU retriever"); common(sp); retrieve_args(sp)
    sp.set_defaults(fn=cmd_retrieve)
    sp = sub.add_parser("answer", help="ask Reactor every question not yet in results"); common(sp); answer_args(sp)
    sp.set_defaults(fn=cmd_answer)
    sp = sub.add_parser("predict", help="results -> predictions.jsonl"); common(sp); predict_args(sp)
    sp.set_defaults(fn=cmd_predict)
    sp = sub.add_parser("score", help="score predictions against gold (dev)"); common(sp, selection=False)
    sp.set_defaults(fn=cmd_score)
    sp = sub.add_parser("run", help="retrieve (if needed) + answer + predict + score")
    common(sp); retrieve_args(sp); answer_args(sp); predict_args(sp)
    sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("chat", help="interactive REPL: type a question, see every retrieval step + Reactor's reasoning")
    sp.add_argument("--split", default="dev", help="dev | test ... (default dev)")
    sp.add_argument("--data", help="directory with the split's files (default: $PIPELINE_DATA_ROOT/<split>)")
    sp.add_argument("--history", help="start on this history id (default: first one)")
    sp.add_argument("--cache", help="embedding cache dir (default retriever/cache/<split>)")
    sp.add_argument("--reranker", default="base", help="base | v2-m3 | none")
    sp.add_argument("--rerank-text", default="best2", choices=["best1", "best2"])
    sp.add_argument("--fusion", default="fused", choices=["fused", "rerank"])
    sp.add_argument("--device", choices=["cuda", "cpu"], help="default: cuda if available")
    provider_args(sp)
    sp.add_argument("--max-chars", type=int, default=None, help=f"default 16000 reactor, {GROQ_DEFAULT_MAX_CHARS} groq")
    sp.add_argument("--attempts", type=int)
    sp.add_argument("--timeout", type=float)
    sp.add_argument("--dry-run", action="store_true", help="show everything but do not call Reactor")
    sp.add_argument("--show-message", action="store_true", help="also print the full message sent to Reactor")
    sp.set_defaults(fn=lambda a: __import__("pipeline.chat", fromlist=["cmd_chat"]).cmd_chat(a))
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    _provider_defaults(a)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
