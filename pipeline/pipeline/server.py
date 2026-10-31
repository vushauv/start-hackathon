"""HTTP bridge for the upstream web chat (``web/``).

``pipeline serve`` streams retrieval traces as NDJSON for the web UI. The independent Dockerized FastAPI
Reactor conversation service is implemented in :mod:`pipeline.reactor_api` and launched with
``pipeline api-serve``. Keeping them separate prevents the web UI's GPU retrieval dependencies from becoming a
requirement for the prebuilt-evidence conversation API.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .answering import Answerer, ReactorAnswerer
from .data import Split, append_jsonl, iter_jsonl
from .models import Prediction, QuestionResult, Retrieved
from .postprocess import Policy, to_prediction
from .retrieval import FileRetrieval, Retrieval
from .runner import answer_one


class AnswerResponse(BaseModel):
    """A client-safe answer plus enough metadata to audit its evidence."""

    question_id: str
    history_id: str
    status: str
    abstain: bool
    answer: str
    citations: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    guessed_type: str = ""
    latency_s: float = 0.0
    error: str = ""


class Conversation(BaseModel):
    """Public, resumable handle for one stateful Reactor history."""

    conversation_id: str
    title: str = ""
    created_at: str
    updated_at: str
    answer_count: int = 0


class ConversationRequest(BaseModel):
    title: str = ""


class ConversationStore:
    """Append-only local state for Reactor conversation handles and their answers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: dict[str, dict] = {}
        self._answers: dict[str, list[AnswerResponse]] = {}
        if self.path.exists():
            for event in iter_jsonl(self.path):
                kind, cid = event.get("kind"), event.get("conversation_id")
                if kind == "conversation" and cid:
                    self._records[cid] = event["conversation"]
                    self._answers.setdefault(cid, [])
                elif kind == "answer" and cid and cid in self._records:
                    self._answers[cid].append(AnswerResponse.model_validate(event["answer"]))
                    self._records[cid]["updated_at"] = event["at"]

    def create(self, reactor_history_id: str, title: str = "") -> Conversation:
        now = _now()
        cid = uuid.uuid4().hex
        record = {"conversation_id": cid, "reactor_history_id": reactor_history_id, "title": title,
                  "created_at": now, "updated_at": now}
        self._records[cid] = record
        self._answers[cid] = []
        append_jsonl(self.path, {"kind": "conversation", "conversation_id": cid, "conversation": record})
        return self.summary(cid)

    def reactor_history_id(self, conversation_id: str) -> str:
        try:
            return self._records[conversation_id]["reactor_history_id"]
        except KeyError as exc:
            raise KeyError(f"unknown conversation_id {conversation_id!r}") from exc

    def update_history(self, conversation_id: str, reactor_history_id: str) -> None:
        record = self._records[conversation_id]
        record["reactor_history_id"] = reactor_history_id
        record["updated_at"] = _now()
        append_jsonl(self.path, {"kind": "conversation", "conversation_id": conversation_id,
                                 "conversation": record})

    def append_answer(self, conversation_id: str, answer: AnswerResponse) -> None:
        if conversation_id not in self._records:
            raise KeyError(f"unknown conversation_id {conversation_id!r}")
        now = _now()
        self._answers[conversation_id].append(answer)
        self._records[conversation_id]["updated_at"] = now
        append_jsonl(self.path, {"kind": "answer", "conversation_id": conversation_id,
                                 "at": now, "answer": answer.model_dump()})

    def summary(self, conversation_id: str) -> Conversation:
        try:
            record = self._records[conversation_id]
        except KeyError as exc:
            raise KeyError(f"unknown conversation_id {conversation_id!r}") from exc
        return Conversation(**{key: record[key] for key in ("conversation_id", "title", "created_at", "updated_at")},
                            answer_count=len(self._answers[conversation_id]))

    def list(self) -> list[Conversation]:
        return sorted((self.summary(cid) for cid in self._records), key=lambda item: item.updated_at, reverse=True)

    def answers(self, conversation_id: str) -> list[AnswerResponse]:
        self.summary(conversation_id)  # validates the id
        return list(self._answers[conversation_id])


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PipelineService:
    """Single-agent service facade. All calls are serialized for Reactor safety."""

    def __init__(self, *, questions: list[dict], retrieval: Retrieval,
                 sessions: Mapping[str, Mapping[str, str]], answerer: Answerer | None = None,
                 max_chars: int = 16_000, policy: Policy | None = None,
                 state_path: str | Path = "conversations.jsonl"):
        self.questions = {q["question_id"]: q for q in questions}
        self.retrieval = retrieval
        self.sessions = sessions
        self.answerer = answerer or ReactorAnswerer()
        self.max_chars = max_chars
        self.policy = policy or Policy()
        self.conversations = ConversationStore(state_path)
        self._lock = threading.Lock()

    @classmethod
    def from_files(cls, *, data_dir: str | Path, evidence_path: str | Path,
                   answerer: Answerer | None = None, max_chars: int = 16_000,
                   state_path: str | Path = "conversations.jsonl") -> "PipelineService":
        split = Split(data_dir)
        return cls(questions=split.questions, retrieval=FileRetrieval(evidence_path), sessions=split.sessions(),
                   answerer=answerer, max_chars=max_chars, state_path=state_path)

    def create_conversation(self, title: str = "") -> Conversation:
        with self._lock:
            reactor_history_id = self.answerer.start_history("conversation")
            return self.conversations.create(reactor_history_id, title)

    def answer_retrieved(self, row: Retrieved) -> AnswerResponse:
        """Ask Reactor using a pre-retrieved question bundle (the generic integration endpoint)."""
        with self._lock:
            reactor_history_id = self.answerer.start_history(row.history_id)
            result = answer_one(row, self.answerer, self.max_chars, reactor_history_id)
        known = set(self.sessions.get(row.history_id, {})) or None
        prediction = to_prediction(result, self.policy, known)
        return _response(result, prediction)

    def answer_question(self, question_id: str) -> AnswerResponse:
        """Retrieve a configured dataset question from the evidence file, then ask Reactor."""
        try:
            question = self.questions[question_id]
        except KeyError as exc:
            raise KeyError(f"unknown question_id {question_id!r}") from exc
        row = self.retrieval.retrieve([question])[0]
        return self.answer_retrieved(row)

    def answer_in_conversation(self, conversation_id: str, question_id: str) -> AnswerResponse:
        """Resume an existing Reactor history to answer a configured dataset question."""
        try:
            question = self.questions[question_id]
        except KeyError as exc:
            raise KeyError(f"unknown question_id {question_id!r}") from exc
        row = self.retrieval.retrieve([question])[0]
        with self._lock:
            reactor_history_id = self.conversations.reactor_history_id(conversation_id)
            # ReactorAnswerer owns this mutable field; restore the selected conversation before asking.
            if hasattr(self.answerer, "history_id"):
                self.answerer.history_id = reactor_history_id
            result = answer_one(row, self.answerer, self.max_chars, reactor_history_id)
            current = getattr(self.answerer, "history_id", None)
            if current:
                self.conversations.update_history(conversation_id, current)
            known = set(self.sessions.get(row.history_id, {})) or None
            answer = _response(result, to_prediction(result, self.policy, known))
            self.conversations.append_answer(conversation_id, answer)
            return answer


def _response(result: QuestionResult, prediction: Prediction) -> AnswerResponse:
    return AnswerResponse(
        question_id=result.question_id,
        history_id=result.history_id,
        status=result.status,
        abstain=prediction.abstain,
        answer=prediction.answer,
        citations=prediction.citations,
        evidence_ids=result.sent_ids,
        guessed_type=result.guessed_type,
        latency_s=result.latency_s,
        error=result.error,
    )


def create_app(service: PipelineService):
    """Build the optional FastAPI app. FastAPI is imported only when serving."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised by the serve command's setup instructions
        raise RuntimeError("FastAPI is not installed; run `uv sync --extra api` first") from exc

    app = FastAPI(title="RxAI Memory Pipeline", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "provider": "reactor", "questions": len(service.questions), "serialized": True}

    @app.post("/v1/dataset/answers/{question_id}", response_model=AnswerResponse)
    def answer_dataset_question(question_id: str) -> AnswerResponse:
        try:
            return service.answer_question(question_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/answers", response_model=AnswerResponse)
    def answer_retrieved(row: Retrieved) -> AnswerResponse:
        return service.answer_retrieved(row)

    @app.post("/v1/conversations", response_model=Conversation)
    def create_conversation(request: ConversationRequest) -> Conversation:
        return service.create_conversation(request.title)

    @app.get("/v1/conversations", response_model=list[Conversation])
    def list_conversations() -> list[Conversation]:
        return service.conversations.list()

    @app.get("/v1/conversations/{conversation_id}", response_model=Conversation)
    def get_conversation(conversation_id: str) -> Conversation:
        try:
            return service.conversations.summary(conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/conversations/{conversation_id}/answers", response_model=list[AnswerResponse])
    def list_answers(conversation_id: str) -> list[AnswerResponse]:
        try:
            return service.conversations.answers(conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/conversations/{conversation_id}/dataset/answers/{question_id}", response_model=AnswerResponse)
    def answer_in_conversation(conversation_id: str, question_id: str) -> AnswerResponse:
        try:
            return service.answer_in_conversation(conversation_id, question_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def serve(*, data_dir: str | Path, evidence_path: str | Path, host: str = "127.0.0.1", port: int = 8000,
          max_chars: int = 16_000, attempts: int = 3, timeout: float | None = None,
          state_path: str | Path = "conversations.jsonl") -> None:
    """Run Uvicorn with Reactor as the only answer provider."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Uvicorn is not installed; run `uv sync --extra api` first") from exc
    answerer = ReactorAnswerer(attempts=attempts, timeout=timeout)
    service = PipelineService.from_files(data_dir=data_dir, evidence_path=evidence_path,
                                         answerer=answerer, max_chars=max_chars, state_path=state_path)
    uvicorn.run(create_app(service), host=host, port=port)
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from .answering import GROQ_DEFAULT_MAX_CHARS, GROQ_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL
from .data import Split, load_histories, split_dir
from .models import Evidence, Retrieved
from .prompting import compose
from .retrieval import RETRIEVER_DIR

INDEX_CACHE = 4  # histories kept indexed in memory
CHANNELS = (("dense_B", "Dense · user turns"), ("dense_F", "Dense · pair windows"), ("bm25", "BM25 · pair windows"))
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _log(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class Engine:
    def __init__(self, a):
        self.a = a
        self.dir = split_dir(a.split, a.data)
        self.split = Split(self.dir)
        self.qs_by_id = {q["question_id"]: q for q in self.split.questions}
        self.by_hid: dict[str, list[dict]] = {}
        for q in self.split.questions:
            self.by_hid.setdefault(q["history_id"], []).append(q)
        self.cache = Path(a.cache) if a.cache else RETRIEVER_DIR / "cache" / self.dir.name
        if str(RETRIEVER_DIR) not in sys.path:
            sys.path.insert(0, str(RETRIEVER_DIR))
        device = a.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        import retriever as R
        self.R = R
        t0 = time.perf_counter()
        self.emb = R.Embedder(device=device)
        if device == "cpu":  # fp16 is slow / partly unsupported on CPU
            self.emb.m.float()
        self.rerankers: dict[str, object] = {}
        self._reranker(a.reranker)
        _log(f"models ready on {device} in {_ms(t0)} ms")
        self.indexes: OrderedDict[str, object] = OrderedDict()
        self.answerers: dict[tuple, object] = {}
        self.reactor_hid_for: dict[tuple, str] = {}
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- resources
    def _reranker(self, name: str):
        if name == "none":
            return None
        if name not in self.rerankers:
            rr = self.R.Reranker(name, device=self.device)
            if self.device == "cpu":
                rr.m.float()
            self.rerankers[name] = rr
        return self.rerankers[name]

    def _index(self, hid: str):
        if hid in self.indexes:
            self.indexes.move_to_end(hid)
            return self.indexes[hid]
        h = load_histories(self.dir / "histories.jsonl", {hid})[hid]
        idx = self.R.build_indexes({hid: h}, self.emb, str(self.cache), log=lambda s: None)[hid]
        self.indexes[hid] = idx
        while len(self.indexes) > INDEX_CACHE:
            self.indexes.popitem(last=False)
        return idx

    def _latest_date(self, idx) -> str:
        dates = [d for d in idx.dates if d]
        return max(dates, key=lambda d: self.R.parse_date(d) or datetime.min.date()) if dates else ""

    # ---------------------------------------------------------------- read endpoints
    def meta(self) -> dict:
        a = self.a
        return {
            "split": a.split, "device": self.device, "has_gold": bool(self.split.gold),
            "embedder": self.R.EMB_MODEL, "rerankers": ["base", "v2-m3", "none"],
            "histories": [{"id": hid, "questions": len(qs), "first_question": qs[0]["question"]}
                          for hid, qs in self.by_hid.items()],
            "defaults": {
                "reranker": a.reranker, "rerank_text": "best2", "fusion": "fused", "rerank_threshold": None,
                "provider": a.provider, "model": a.model or "", "max_chars": a.max_chars,
                "attempts": None, "timeout": None, "dry_run": a.dry_run, "show_message": False,
            },
            "provider_defaults": {
                "reactor": {"model": "", "max_chars": 16000},
                "groq": {"model": GROQ_DEFAULT_MODEL, "max_chars": GROQ_DEFAULT_MAX_CHARS},
                "openai": {"model": OPENAI_DEFAULT_MODEL, "max_chars": GROQ_DEFAULT_MAX_CHARS},
            },
        }

    def history(self, hid: str) -> dict:
        if hid not in self.by_hid:
            raise KeyError(f"unknown history {hid}")
        t0 = time.perf_counter()
        with self.lock:
            idx = self._index(hid)
        qs = self.by_hid[hid]
        return {
            "id": hid, "sessions": len(idx.sids), "b_chunks": len(idx.B_emb), "f_chunks": len(idx.F_text),
            "question_date": self._latest_date(idx), "index_ms": _ms(t0),
            "questions": [{"id": q["question_id"], "question": q["question"], "date": q.get("question_date", ""),
                           "type": self.split.gold.get(q["question_id"], {}).get("question_type", "")}
                          for q in qs],
        }

    # ---------------------------------------------------------------- ask
    def ask(self, req: dict) -> Iterator[dict]:
        if not self.lock.acquire(blocking=False):
            yield {"type": "queued"}
            self.lock.acquire()
        try:
            yield from self._ask(req)
        except Exception as e:  # keep the server alive; the client shows the error
            _log(f"ask failed: {type(e).__name__}: {e}")
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        finally:
            self.lock.release()

    def _settings(self, s: dict) -> dict:
        a = self.a
        out = {
            "reranker": s.get("reranker") or a.reranker,
            "rerank_text": s.get("rerank_text") or "best2",
            "fusion": s.get("fusion") or "fused",
            "rerank_threshold": s.get("rerank_threshold"),
            "provider": s.get("provider") or a.provider,
            "model": s.get("model") or None,
            "max_chars": s.get("max_chars") or None,
            "attempts": s.get("attempts") or None,
            "timeout": s.get("timeout") or None,
            "dry_run": bool(s.get("dry_run", a.dry_run)),
            "show_message": bool(s.get("show_message", False)),
        }
        api = out["provider"] in ("groq", "openai")
        if not out["model"] and api:
            out["model"] = OPENAI_DEFAULT_MODEL if out["provider"] == "openai" else GROQ_DEFAULT_MODEL
        if not out["max_chars"]:
            out["max_chars"] = GROQ_DEFAULT_MAX_CHARS if api else 16000
        return out

    def _ask(self, req: dict) -> Iterator[dict]:
        R = self.R
        s = self._settings(req.get("settings") or {})
        qmeta = self.qs_by_id.get(req.get("question_id") or "")
        hid = qmeta["history_id"] if qmeta else req.get("history_id")
        if hid not in self.by_hid:
            raise KeyError(f"unknown history {hid}")
        question = (req.get("question") or (qmeta or {}).get("question") or "").strip()
        if not question:
            raise ValueError("empty question")
        T0 = time.perf_counter()
        idx = self._index(hid)
        qdate = (req.get("question_date") or "").strip()
        if _ISO.match(qdate):
            qdate = datetime.strptime(qdate, "%Y-%m-%d").strftime("%Y/%m/%d (%a) 23:59")
        qdate = qdate or (qmeta or {}).get("question_date") or self._latest_date(idx)
        gold = self.split.gold.get(qmeta["question_id"]) if qmeta else None
        gold_ev = set(gold.get("evidence", [])) if gold else set()
        sess = lambda si: {"sid": idx.sids[si], "date": idx.dates[si], "gold": idx.sids[si] in gold_ev}
        yield {"type": "start", "history_id": hid, "question": question, "question_date": qdate,
               "question_id": qmeta["question_id"] if qmeta else None, "settings": s}

        # 1. embedding
        t0 = time.perf_counter()
        qv = self.emb.encode_queries([question])[0]
        yield {"type": "step", "id": "embedding", "ms": _ms(t0), "data": {
            "model": R.EMB_MODEL, "prefix": R.BGE_PREFIX.strip(), "dim": int(qv.shape[0]),
            "norm": float(np.linalg.norm(qv)), "head": [float(x) for x in qv[:8]]}}

        # 2. first stage
        t0 = time.perf_counter()
        tr: dict = {}
        cands = R.first_stage(idx, qv, question, trace=tr)
        ranks, channels = {}, []
        for name, label in CHANNELS:
            sc = tr[name]
            order = [int(i) for i in np.argsort(-sc, kind="stable") if np.isfinite(sc[i])]
            ranks[name] = {si: r for r, si in enumerate(order)}
            channels.append({"name": name, "label": label,
                             "top": [{**sess(si), "score": float(sc[si])} for si in order[:10]]})
        yield {"type": "step", "id": "first_stage", "ms": _ms(t0), "data": {
            "rrf_k": R.RRF_K, "channels": channels,
            "candidates": [{**sess(c["si"]), "rrf": c["rrf"],
                            "ranks": {n: (ranks[n][c["si"]] + 1 if c["si"] in ranks[n] else None) for n, _ in CHANNELS}}
                           for c in cands]}}

        # 3. rerank
        t0 = time.perf_counter()
        rr = self._reranker(s["reranker"])
        if rr:
            texts = [R.rerank_text(idx, c, s["rerank_text"]) for c in cands]
            scores = rr.score([question] * len(cands), texts)
            ordered = R.final_order(cands, scores, s["fusion"])
            rr_rank = {cands[i]["sid"]: r for r, i in enumerate(np.argsort(-scores, kind="stable"))}
            rr_score = {c["sid"]: float(x) for c, x in zip(cands, scores)}
        else:
            ordered, rr_rank, rr_score = list(cands), None, None
        gt = R.guess_type(question)
        thr = s["rerank_threshold"]
        chosen = R.select_evidence(ordered, rr_score, gt, None if thr in (None, "") else float(thr))
        chosen_ids = {c["sid"] for c in chosen}
        rrf_pos = {c["sid"]: i for i, c in enumerate(cands)}
        yield {"type": "step", "id": "rerank", "ms": _ms(t0), "data": {
            "reranker": rr.name if rr else "none", "fusion": s["fusion"], "rerank_text": s["rerank_text"],
            "order": [{**sess(c["si"]), "score": rr_score[c["sid"]] if rr_score else None,
                       "rerank_rank": rr_rank[c["sid"]] + 1 if rr_rank else None,
                       "rrf_rank": rrf_pos[c["sid"]] + 1, "moved": rrf_pos[c["sid"]] - r,
                       "selected": c["sid"] in chosen_ids} for r, c in enumerate(ordered)]}}

        # 4. type / k / hint
        ev = R.pack_evidence(idx, chosen)
        hint = R.make_hint(gt, qdate, ev)
        yield {"type": "step", "id": "selection", "ms": 0, "data": {
            "guessed_type": gt, "gold_type": gold.get("question_type") if gold else None,
            "k": len(chosen), "rule": f"threshold ≥ {thr}" if thr not in (None, "") else f"top-{R.topk_for(gt)} for type",
            "question_date": qdate, "hint": hint}}

        # 5. evidence
        yield {"type": "step", "id": "evidence", "ms": 0, "data": {
            "chars": sum(len(e["text"]) for e in ev), "cap": R.EVIDENCE_CHARS,
            "passages": [{"label": f"E{i + 1}", "id": e["id"], "date": e["date"], "chars": len(e["text"]),
                          "text": e["text"], "gold": e["id"] in gold_ev} for i, e in enumerate(ev)]}}

        # 6. compose
        r = Retrieved(question_id=qmeta["question_id"] if qmeta else f"chat-{int(time.time())}",
                      history_id=hid, question=question, question_date=qdate, guessed_type=gt,
                      evidence=[Evidence(**e) for e in ev], hint=hint, candidates=[c["sid"] for c in ordered])
        comp = compose(r, s["max_chars"])
        yield {"type": "step", "id": "compose", "ms": 0, "data": {
            "chars": comp.chars, "limit": s["max_chars"], "passages": len(comp.evidence),
            "notes_chars": len(comp.notes), "retrieval_ms": _ms(T0),
            "message": comp.message if s["show_message"] else None}}

        # 7. answer
        who = f"{s['provider']} · {s['model']}" if s["provider"] != "reactor" else "Reactor"
        if s["dry_run"]:
            yield {"type": "answer", "status": "dry-run", "who": who, "answer": "", "citations": []}
        else:
            yield {"type": "waiting", "who": who,
                   "expect": "a few seconds" if s["provider"] != "reactor" else "typically 40–120 s"}
            yield self._answer(r, comp, s, who)
        if gold:
            yield {"type": "gold", "abstain": bool(gold.get("abstain")), "answer": str(gold.get("answer", "")),
                   "evidence": gold.get("evidence", []), "question_type": gold.get("question_type")}
        yield {"type": "done", "total_ms": _ms(T0)}

    def _answer(self, r: Retrieved, comp, s: dict, who: str) -> dict:
        from reactor_service import NoAnswer
        from .answering import make_answerer
        key = (s["provider"], s["model"], s["attempts"], s["timeout"])
        if key not in self.answerers:
            self.answerers[key] = make_answerer(s["provider"], model=s["model"], attempts=s["attempts"],
                                                timeout=s["timeout"], log=_log)
        answerer = self.answerers[key]
        t0 = time.perf_counter()
        try:
            if self.reactor_hid_for.get(key) != r.history_id:
                answerer.start_history(r.history_id)
                self.reactor_hid_for[key] = r.history_id
            a, attempts = answerer.ask(r, comp)
        except NoAnswer as e:
            return {"type": "answer", "status": "no-answer", "who": who, "answer": "", "citations": [],
                    "error": f"{type(e).__name__}: {e}", "latency_s": (time.perf_counter() - t0)}
        dates = {e.id: e.date for e in comp.evidence}
        return {"type": "answer", "status": "abstained" if a.abstained else "answered", "who": who,
                "answer": a.answer or "", "reply": a.reply or "", "latency_s": a.latency_s, "attempts": attempts,
                "reasoning": getattr(answerer, "last_reasoning", "") or "",
                "citations": [{"label": c.label, "id": c.id, "date": dates.get(c.id, "")} for c in a.citations]}


def _handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter than the default per-request stderr line
            pass

        def _head(self, code: int, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _json(self, code: int, body: dict) -> None:
            self._head(code)
            self.wfile.write(json.dumps(body, ensure_ascii=False).encode())

        def do_OPTIONS(self):
            self._head(204)

        def do_GET(self):
            try:
                if self.path == "/api/meta":
                    return self._json(200, engine.meta())
                if self.path.startswith("/api/history/"):
                    return self._json(200, engine.history(self.path.rsplit("/", 1)[1]))
                self._json(404, {"error": "not found"})
            except KeyError as e:
                self._json(404, {"error": str(e).strip("'")})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            if self.path != "/api/ask":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError as e:
                return self._json(400, {"error": f"bad JSON: {e}"})
            self._head(200, "application/x-ndjson")
            try:
                with closing(engine.ask(req)) as events:  # closing releases the engine lock on disconnect
                    for ev in events:
                        self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode())
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser went away

    return H


def cmd_serve(a) -> int:
    engine = Engine(a)
    srv = ThreadingHTTPServer((a.host, a.port), _handler(engine))
    _log(f"pipeline API on http://{a.host}:{a.port}  (split {a.split}, {len(engine.by_hid)} histories)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
