"""``pipeline ui``: a small web app to run questions one at a time and look at what Reactor gets and says.

    uv sync --extra ui && uv run pipeline ui --split test          # http://127.0.0.1:8000
    PIPELINE_UI_DRY_RUN=1 uv run pipeline ui --split dev           # no Reactor: fake answers, for trying the UI

It reuses the pipeline's stages unchanged: ``FileRetrieval`` over ``runs/<split>/evidence.jsonl``, ``compose``,
``ReactorAnswerer`` (one Reactor history per dataset history, kept across clicks), ``answer_one`` and the same
``results.jsonl``, so a question answered here counts as done for ``pipeline answer`` and vice versa.
Custom questions (typed, not from questions.jsonl) need the GPU retriever (``--extra retrieval``); the server
loads it on first use and reports when it is not available.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from reactor_service import ConfigError, ReactorError

from .answering import Answerer, DryRunAnswerer, ReactorAnswerer
from .data import Split, append_jsonl, data_root, iter_jsonl, split_dir
from .models import QuestionResult, Retrieved
from .postprocess import Policy, to_prediction
from .prompting import compose
from .retrieval import FileRetrieval, LocalRetrieval, Retrieval
from .runner import answer_one, write_predictions

UI_DIR = Path(__file__).resolve().parent / "ui"
SPLIT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def check_split(name: str) -> str:
    """Split names become directory names under runs/ and data/: letters, digits, '-' and '_' only."""
    if not SPLIT_NAME.match(name):
        raise HTTPException(400, f"invalid split name {name!r}")
    return name


class AskBody(BaseModel):
    split: str
    question_id: str
    max_chars: int = 16000


class CustomBody(BaseModel):
    split: str
    history_id: str
    question: str
    question_date: str = ""
    max_chars: int = 16000


class PredictBody(BaseModel):
    split: str
    fallback_citations: int = 1
    abstain_citations: int = 1


class State:
    """Everything the server keeps between requests. One Reactor conversation at a time (``lock``)."""

    def __init__(self, runs: Path, data: str | None = None, dry_run: bool = False, attempts: int = 3):
        self.runs, self.data, self.dry_run, self.attempts = runs, data, dry_run, attempts
        self.lock = threading.Lock()
        self.splits: dict[str, Split] = {}
        self.evidence: dict[str, FileRetrieval] = {}
        self.answerer: Answerer | None = None
        self.dataset_history: str | None = None  # the dataset history the current Reactor history is for
        self.reactor_history: str = ""
        self.local: dict[str, Retrieval] = {}

    def split(self, name: str) -> Split:
        check_split(name)
        if name not in self.splits:
            self.splits[name] = Split(split_dir(name, self.data))
        return self.splits[name]

    def paths(self, name: str) -> dict[str, Path]:
        d = self.runs / check_split(name)
        return {k: d / f"{k}.jsonl" for k in ("evidence", "results", "predictions", "score_details")}

    def retrieval(self, name: str) -> FileRetrieval:
        p = self.paths(name)["evidence"]
        if name not in self.evidence or self.evidence[name].path.stat().st_mtime != getattr(self, f"_mt_{name}", None):
            if not p.exists():
                raise HTTPException(404, f"{p} not found: run `pipeline retrieve --split {name}` on the GPU box first")
            self.evidence[name] = FileRetrieval(p)
            setattr(self, f"_mt_{name}", p.stat().st_mtime)
        return self.evidence[name]

    def local_retrieval(self, name: str) -> Retrieval:
        if name not in self.local:
            try:
                self.local[name] = LocalRetrieval(split_dir(name, self.data), log=lambda s: None)
            except ImportError as e:
                raise HTTPException(503, "custom questions need the GPU retriever: `uv sync --extra retrieval` "
                                         f"on a CUDA box ({e})") from None
        return self.local[name]

    def get_answerer(self) -> Answerer:
        if self.answerer is None:
            if self.dry_run:
                self.answerer = DryRunAnswerer()
            else:
                try:
                    self.answerer = ReactorAnswerer(attempts=self.attempts)
                except ConfigError as e:
                    raise HTTPException(503, str(e)) from None
        return self.answerer

    def reactor_history_for(self, dataset_history_id: str) -> str:
        """Reuse the Reactor history while we stay in the same dataset history, else start a fresh one."""
        a = self.get_answerer()
        if self.dataset_history != dataset_history_id or not self.reactor_history:
            try:
                self.reactor_history = a.start_history(dataset_history_id)
            except ReactorError as e:
                raise HTTPException(502, f"could not start a Reactor history: {e}") from None
            self.dataset_history = dataset_history_id
        # a ReactorAnswerer may have had to restart its history on `history_changed`
        self.reactor_history = getattr(a, "history_id", None) or self.reactor_history
        return self.reactor_history

    def results(self, name: str) -> dict[str, dict]:
        p = self.paths(name)["results"]
        out: dict[str, dict] = {}
        if p.exists():
            for r in iter_jsonl(p):
                out[r["question_id"]] = r  # last one wins, as in write_predictions
        return out


def create_app(state: State, allowed_hosts: list[str] | None = None) -> FastAPI:
    from fastapi.middleware.trustedhost import TrustedHostMiddleware

    from urllib.parse import urlsplit

    from fastapi import Request
    from fastapi.responses import JSONResponse

    hosts = allowed_hosts or ["127.0.0.1", "localhost"]
    app = FastAPI(title="memory pipeline UI")
    # Only answer requests addressed to the host we were started on: blocks DNS-rebinding from a web page.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.middleware("http")
    async def same_origin_posts(request: Request, call_next):
        """A POST from another web page's origin (CSRF) is refused; our page sends no Origin or our own."""
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and (urlsplit(origin).hostname or "") not in hosts:
                return JSONResponse({"detail": f"cross-origin request from {origin} refused"}, status_code=403)
            if request.headers.get("sec-fetch-site") in ("cross-site", "same-site"):
                return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(UI_DIR / "index.html")

    @app.get("/api/splits")
    def splits():
        root = data_root()
        names = sorted(p.name for p in root.iterdir() if (p / "questions.jsonl").exists()) if root.exists() else []
        try:
            split_dir("dev", state.data)
            names = names or ["dev"]
        except FileNotFoundError:
            pass
        return {"splits": names, "runs": str(state.runs), "dry_run": state.dry_run}

    @app.get("/api/questions")
    def questions(split: str):
        try:
            sp = state.split(split)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from None
        res = state.results(split)
        have_ev = state.paths(split)["evidence"].exists()
        rows = []
        for q in sp.questions:
            g = sp.gold.get(q["question_id"])
            r = res.get(q["question_id"])
            rows.append({**q,
                         "gold": {k: g[k] for k in ("question_type", "abstain", "answer", "evidence")} if g else None,
                         "result": {k: r.get(k) for k in ("status", "answer_raw", "cited_ids", "latency_s")} if r else None})
        return {"questions": rows, "has_evidence": have_ev, "has_gold": bool(sp.gold)}

    @app.get("/api/question")
    def question(split: str, question_id: str, max_chars: int = 16000):
        sp = state.split(split)
        q = next((x for x in sp.questions if x["question_id"] == question_id), None)
        if q is None:
            raise HTTPException(404, f"unknown question {question_id}")
        r = state.retrieval(split).retrieve([q])[0]
        c = compose(r, max_chars)
        res = state.results(split).get(question_id)
        pred = to_prediction(QuestionResult.model_validate(res)).model_dump() if res else None
        return {"question": q, "gold": sp.gold.get(question_id), "retrieved": r.model_dump(),
                "composed": {"evidence_ids": [e.id for e in c.evidence], "notes": c.notes, "message": c.message,
                             "chars": c.chars, "max_chars": max_chars},
                "result": res, "prediction": pred}

    def _ask(split: str, r: Retrieved, max_chars: int, save: bool) -> dict:
        if not state.lock.acquire(blocking=False):
            raise HTTPException(409, "another question is being asked; Reactor takes one at a time")
        try:
            rh = state.reactor_history_for(r.history_id)
            res = answer_one(r, state.get_answerer(), max_chars, rh)
            if save:
                append_jsonl(state.paths(split)["results"], res.model_dump())
            known = set(state.split(split).sessions().get(r.history_id, {})) or None
            return {"result": res.model_dump(), "prediction": to_prediction(res, Policy(), known).model_dump()}
        finally:
            state.lock.release()

    @app.post("/api/ask")
    def ask(body: AskBody):
        sp = state.split(body.split)
        q = next((x for x in sp.questions if x["question_id"] == body.question_id), None)
        if q is None:
            raise HTTPException(404, f"unknown question {body.question_id}")
        r = state.retrieval(body.split).retrieve([q])[0]
        return _ask(body.split, r, body.max_chars, save=True)

    @app.post("/api/ask-custom")
    def ask_custom(body: CustomBody):
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        q = {"question_id": "custom", "history_id": body.history_id, "question": body.question.strip(),
             "question_date": body.question_date}
        r = state.local_retrieval(body.split).retrieve([q])[0]
        out = _ask(body.split, r, body.max_chars, save=False)
        c = compose(r, body.max_chars)
        out["retrieved"] = r.model_dump()
        out["composed"] = {"message": c.message, "chars": c.chars}
        return out

    @app.post("/api/predict")
    def predict(body: PredictBody):
        sp = state.split(body.split)
        p = state.paths(body.split)
        policy = Policy(fallback_citations=body.fallback_citations, abstain_citations=body.abstain_citations)
        log: list[str] = []
        preds = write_predictions(p["results"], p["predictions"], questions=sp.questions, policy=policy,
                                  sessions=sp.sessions(), log=log.append)
        out = {"predictions": str(p["predictions"]), "n": len(preds), "abstained": sum(x.abstain for x in preds),
               "log": log, "score": None}
        gold = sp.dir / "gold.jsonl"
        if gold.exists():
            from .scoring import score
            out["score"] = score(gold, p["predictions"], p["score_details"])
        return out

    return app


def serve(split: str | None, *, host: str = "127.0.0.1", port: int = 8000, runs: Path, data: str | None = None,
          dry_run: bool = False, attempts: int = 3) -> None:
    import uvicorn

    state = State(runs, data, dry_run=dry_run or os.environ.get("PIPELINE_UI_DRY_RUN") == "1", attempts=attempts)
    hosts = ["127.0.0.1", "localhost"] if host in ("127.0.0.1", "localhost") else [host, "127.0.0.1", "localhost"]
    if host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: binding to {host}: anyone who can reach this port can ask Reactor with your key. "
              "Prefer an SSH tunnel to 127.0.0.1.", flush=True)
    app = create_app(state, allowed_hosts=hosts)
    app.state.default_split = split
    uvicorn.run(app, host=host, port=port, log_level="info")
