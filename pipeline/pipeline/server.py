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
import queue
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from reactor_service import ConfigError, ReactorError

from .answering import GROQ_DEFAULT_MAX_CHARS, GROQ_DEFAULT_MODEL, Answerer, DryRunAnswerer, make_answerer
from .data import Split, append_jsonl, data_root, iter_jsonl, load_histories, split_dir
from .models import QuestionResult, Retrieved
from .postprocess import Policy, strip_labels, to_prediction
from .quotes import locate
from .prompting import compose
from .retrieval import FileRetrieval, LocalRetrieval, Retrieval
from .runner import answer_one, write_predictions
from .sessions_db import DBUnavailable, SessionsDB

UI_DIR = Path(__file__).resolve().parent / "ui"


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


_NAME = re.compile(r"\b(?:Hi|Hey|Hello|Of course|Sure|Thanks|Thank you|Great|Got it|Perfect|Absolutely),? ([A-Z][a-z]{2,})[!,.]")
_NOT_NAMES = {"There", "That", "This", "Sure", "Great", "Absolutely", "Thanks", "Perfect", "Here", "What", "Yes",
              "Okay", "Certainly", "Definitely", "Good", "Nice", "Glad", "Sounds", "Congratulations"}


def guess_name(history: dict) -> str | None:
    """The user's first name, if the assistant addresses them by it at least twice."""
    c: Counter[str] = Counter()
    for s in history.get("sessions", []):
        for t in s.get("turns", []):
            if t.get("role") == "assistant":
                for m in _NAME.finditer(t.get("content") or ""):
                    if m.group(1) not in _NOT_NAMES:
                        c[m.group(1)] += 1
    if not c:
        return None
    name, n = c.most_common(1)[0]
    return name if n >= 2 else None


class Job:
    def __init__(self, body: dict):
        self.id = uuid.uuid4().hex[:12]
        self.body = body
        self.stage = "queued"  # queued | retrieving | answering | locating | done | error
        self.started = time.time()
        self.finished: float | None = None
        self.result: dict | None = None
        self.error: str | None = None

    def view(self) -> dict:
        end = self.finished or time.time()
        return {"job_id": self.id, "stage": self.stage, "started": self.started, "elapsed_s": round(end - self.started, 1),
                "result": self.result, "error": self.error}


SPLIT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def check_split(name: str) -> str:
    """A split name is used in file paths: letters, digits, ``_`` and ``-`` only (no ``..``, no separators)."""
    if not isinstance(name, str) or not SPLIT_NAME.match(name):
        raise HTTPException(400, "bad split name")
    return name


class State:
    """Everything the server keeps between requests. One Reactor conversation at a time (``lock``)."""

    def __init__(self, runs: Path, data: str | None = None, dry_run: bool = False, attempts: int = 3, *,
                 provider: str = "reactor", model: str | None = None, device: str = "auto"):
        self.runs, self.data, self.dry_run, self.attempts = runs, data, dry_run, attempts
        self.provider, self.device = provider, device
        self.model = model or (GROQ_DEFAULT_MODEL if provider == "groq" else None)
        self.lock = threading.Lock()
        self.splits: dict[str, Split] = {}
        self.evidence: dict[str, FileRetrieval] = {}
        self.answerers: dict[str, Answerer] = {}
        self.hist_for: dict[str, tuple[str, str]] = {}  # provider -> (dataset history, answerer history)
        self.local: dict[str, Retrieval] = {}
        # web app: histories (small LRU), per-split people summary, jobs
        self.histories: dict[tuple[str, str], dict] = {}
        self.people_cache: dict[str, list[dict]] = {}
        self.jobs: dict[str, Job] = {}
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.db = SessionsDB()  # sessions and transcripts from PostgreSQL when it is up
        self.run_jobs_inline = False  # tests: run a job inside the request

    @property
    def dataset_history(self) -> str | None:  # kept for the old runner page and tests
        return self.hist_for.get(self.provider, (None, ""))[0]

    def max_chars(self, provider: str) -> int:
        return GROQ_DEFAULT_MAX_CHARS if provider == "groq" else 16000

    def split(self, name: str) -> Split:
        check_split(name)
        if name not in self.splits:
            self.splits[name] = Split(split_dir(name, self.data))
        return self.splits[name]

    def paths(self, name: str) -> dict[str, Path]:
        check_split(name)
        d = (self.runs / name).resolve()
        if not d.is_relative_to(self.runs.resolve()):
            raise HTTPException(400, "bad split name")
        return {k: d / f"{k}.jsonl" for k in ("evidence", "results", "predictions", "score_details")}

    def retrieval(self, name: str) -> FileRetrieval:
        check_split(name)
        p = self.paths(name)["evidence"]
        if name not in self.evidence or self.evidence[name].path.stat().st_mtime != getattr(self, f"_mt_{name}", None):
            if not p.exists():
                raise HTTPException(404, f"{p} not found: run `pipeline retrieve --split {name}` on the GPU box first")
            self.evidence[name] = FileRetrieval(p)
            setattr(self, f"_mt_{name}", p.stat().st_mtime)
        return self.evidence[name]

    def local_retrieval(self, name: str) -> Retrieval:
        check_split(name)
        if name not in self.local:
            try:
                self.local[name] = LocalRetrieval(split_dir(name, self.data), device=self.device, log=lambda s: None)
            except ImportError as e:
                raise HTTPException(503, "live questions need the retriever: `uv sync --extra retrieval --extra ui` "
                                         f"({e})") from None
        return self.local[name]

    def get_answerer(self, provider: str | None = None) -> Answerer:
        provider = provider or self.provider
        if provider not in self.answerers:
            if self.dry_run:
                self.answerers[provider] = DryRunAnswerer()
            else:
                try:
                    self.answerers[provider] = make_answerer(
                        provider, model=self.model if provider == "groq" else None, attempts=self.attempts)
                except (ConfigError, ValueError, OSError, KeyError) as e:
                    raise HTTPException(503, f"{provider}: {e}") from None
        return self.answerers[provider]

    def reactor_history_for(self, dataset_history_id: str, provider: str | None = None) -> str:
        """Reuse the answerer's history while we stay in the same dataset history, else start a fresh one."""
        provider = provider or self.provider
        a = self.get_answerer(provider)
        cur = self.hist_for.get(provider)
        if cur is None or cur[0] != dataset_history_id or not cur[1]:
            try:
                rh = a.start_history(dataset_history_id)
            except ReactorError as e:
                raise HTTPException(502, f"could not start a Reactor history: {e}") from None
            cur = (dataset_history_id, rh)
        # a ReactorAnswerer may have had to restart its history on `history_changed`
        self.hist_for[provider] = (cur[0], getattr(a, "history_id", None) or cur[1])
        return self.hist_for[provider][1]

    # ------------------------------------------------------------------ web app data
    def history(self, name: str, hid: str) -> dict:
        check_split(name)
        key = (name, hid)
        if key not in self.histories:
            h = load_histories(self.split(name).dir / "histories.jsonl", {hid}).get(hid)
            if h is None:
                raise HTTPException(404, f"unknown history {hid}")
            if len(self.histories) >= 8:
                self.histories.pop(next(iter(self.histories)))
            self.histories[key] = h
        return self.histories[key]

    def people(self, name: str) -> list[dict]:
        check_split(name)
        if name not in self.people_cache:
            sp = self.split(name)
            by_h: dict[str, list[dict]] = {}
            for q in sp.questions:
                by_h.setdefault(q["history_id"], []).append(
                    {k: q.get(k, "") for k in ("question_id", "question", "question_date")})
            out = []
            for hid, h in load_histories(sp.dir / "histories.jsonl", set(by_h)).items():
                dates = sorted(d for d in (s.get("date", "") for s in h["sessions"]) if d)
                out.append({"history_id": hid, "n_sessions": len(h["sessions"]), "first_date": dates[0] if dates else "",
                            "last_date": dates[-1] if dates else "", "name": guess_name(h), "questions": by_h[hid]})
            out.sort(key=lambda p: (p["name"] or "~", p["history_id"]))
            self.people_cache[name] = out
        return self.people_cache[name]

    def results(self, name: str) -> dict[str, dict]:
        p = self.paths(name)["results"]
        out: dict[str, dict] = {}
        if p.exists():
            for r in iter_jsonl(p):
                out[r["question_id"]] = r  # last one wins, as in write_predictions
        return out


def create_app(state: State) -> FastAPI:
    app = FastAPI(title="memory pipeline UI")

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

    # ------------------------------------------------------------------ the reading-room web app
    def _session(split: str, hid: str, sid: str) -> dict:
        h = state.history(split, hid)
        s = next((x for x in h["sessions"] if x["session_id"] == sid), None)
        if s is None:
            raise HTTPException(404, f"unknown session {sid}")
        return s

    @app.get("/api/meta")
    def meta():
        return {"provider": state.provider, "model": state.model, "dry_run": state.dry_run, "device": state.device,
                "providers": ["reactor", "groq"], "max_chars": {p: state.max_chars(p) for p in ("reactor", "groq")},
                "default_split": getattr(app.state, "default_split", None)}

    @app.get("/api/people")
    def people(split: str):
        try:
            return {"split": split, "people": state.people(split)}
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from None

    @app.get("/api/history")
    def history(split: str, history_id: str):
        try:
            return {"history_id": history_id, "sessions": state.db.sessions(split, history_id), "source": "postgres"}
        except DBUnavailable:
            pass  # no database: read the dataset files
        h = state.history(split, history_id)
        sessions = []
        for s in h["sessions"]:
            first = next((t["content"] for t in s["turns"] if t["role"] == "user" and t.get("content")), "")
            sessions.append({"session_id": s["session_id"], "date": s.get("date", ""), "n_turns": len(s["turns"]),
                             "preview": first[:140]})
        return {"history_id": history_id, "sessions": sessions, "source": "files"}

    @app.get("/api/session")
    def session(split: str, history_id: str, session_id: str):
        try:
            found = state.db.session(split, history_id, session_id)
        except DBUnavailable:
            found = None
        else:
            if found is None:
                raise HTTPException(404, f"unknown session {session_id}")
            return found
        s = _session(split, history_id, session_id)
        h = state.history(split, history_id)
        ids = [x["session_id"] for x in h["sessions"]]
        i = ids.index(session_id)
        return {"history_id": history_id, "session_id": session_id, "date": s.get("date", ""), "turns": s["turns"],
                "prev": ids[i - 1] if i > 0 else None, "next": ids[i + 1] if i + 1 < len(ids) else None,
                "index": i, "total": len(ids)}

    def _run_job(job: Job) -> None:
        b = job.body
        split, hid, provider = b["split"], b["history_id"], b.get("provider") or state.provider
        try:
            h = state.history(split, hid)
            dates = [x.get("date", "") for x in h["sessions"] if x.get("date")]
            qdate = b.get("question_date") or (max(dates) if dates else "")
            q = {"question_id": f"ui-{job.id}", "history_id": hid, "question": b["question"].strip(),
                 "question_date": qdate}
            job.stage = "retrieving"
            r = state.local_retrieval(split).retrieve([q])[0]
            job.stage = "answering"
            with state.lock:
                rh = state.reactor_history_for(hid, provider)
                res = answer_one(r, state.get_answerer(provider), b.get("max_chars") or state.max_chars(provider), rh)
            job.stage = "locating"
            if res.status == "no-answer":
                raise RuntimeError(res.error or "no answer")
            known = {x["session_id"] for x in h["sessions"]}
            pred = to_prediction(res, Policy(), known)
            answer = pred.answer if not pred.abstain else ""
            cites = res.cited_ids if pred.abstain else pred.citations
            text_of = {e.id: e.text for e in r.evidence}
            sess = {x["session_id"]: x for x in h["sessions"]}
            citations = []
            for sid in dict.fromkeys(c for c in cites if c in sess):
                label = f"E{res.sent_ids.index(sid) + 1}" if sid in res.sent_ids else ""
                quotes = locate(sess[sid], q["question"], answer or q["question"], text_of.get(sid, ""))
                citations.append({"label": label, "session_id": sid, "date": sess[sid].get("date", ""),
                                  "quotes": [x.to_dict() for x in quotes]})
            job.result = {"question": q["question"], "question_date": qdate, "answer": answer,
                          "abstained": pred.abstain, "answer_raw": res.answer_raw, "provider": provider,
                          "latency_s": res.latency_s, "guessed_type": res.guessed_type, "citations": citations,
                          "sent": [{"session_id": e.id, "date": e.date} for e in r.evidence]}
            job.stage = "done"
        except HTTPException as e:
            job.error, job.stage = str(e.detail), "error"
        except Exception as e:  # noqa: BLE001  (the job reports it; the worker must survive)
            job.error, job.stage = f"{type(e).__name__}: {e}", "error"
        finally:
            job.finished = time.time()

    def _worker() -> None:
        while True:
            _run_job(state.job_queue.get())

    @app.post("/api/jobs", status_code=202)
    def create_job(body: dict):
        check_split(str(body.get("split") or ""))
        for k in ("split", "history_id", "question"):
            if not str(body.get(k) or "").strip():
                raise HTTPException(400, f"missing {k}")
        if body.get("provider") not in (None, "reactor", "groq"):
            raise HTTPException(400, "provider must be reactor or groq")
        job = Job(body)
        for old in list(state.jobs)[:-50]:
            state.jobs.pop(old, None)
        state.jobs[job.id] = job
        if state.run_jobs_inline:
            _run_job(job)
        else:
            if state.worker is None:
                state.worker = threading.Thread(target=_worker, daemon=True, name="ask-worker")
                state.worker.start()
            state.job_queue.put(job)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job.view()

    return app


def serve(split: str | None, *, host: str = "127.0.0.1", port: int = 8001, runs: Path, data: str | None = None,
          dry_run: bool = False, attempts: int = 3, provider: str = "reactor", model: str | None = None,
          device: str = "auto") -> None:
    import uvicorn

    state = State(runs, data, dry_run=dry_run or os.environ.get("PIPELINE_UI_DRY_RUN") == "1", attempts=attempts,
                  provider=provider, model=model, device=device)
    app = create_app(state)
    app.state.default_split = split
    uvicorn.run(app, host=host, port=port, log_level="info")
