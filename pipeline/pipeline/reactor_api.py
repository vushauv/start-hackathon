"""FastAPI adapter for the prebuilt-evidence -> Reactor answering pipeline."""

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
    conversation_id: str
    title: str = ""
    created_at: str
    updated_at: str
    answer_count: int = 0


class ConversationRequest(BaseModel):
    title: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConversationStore:
    """Append-only local state; only public metadata and answer records are exposed."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        self.saved_answers: dict[str, list[AnswerResponse]] = {}
        if self.path.exists():
            for event in iter_jsonl(self.path):
                cid = event.get("conversation_id")
                if event.get("kind") == "conversation" and cid:
                    self.records[cid] = event["conversation"]
                    self.saved_answers.setdefault(cid, [])
                elif event.get("kind") == "answer" and cid in self.records:
                    self.saved_answers[cid].append(AnswerResponse.model_validate(event["answer"]))
                    self.records[cid]["updated_at"] = event["at"]

    def create(self, history_id: str, title: str) -> Conversation:
        now, cid = _now(), uuid.uuid4().hex
        record = {"conversation_id": cid, "reactor_history_id": history_id, "title": title,
                  "created_at": now, "updated_at": now}
        self.records[cid], self.saved_answers[cid] = record, []
        append_jsonl(self.path, {"kind": "conversation", "conversation_id": cid, "conversation": record})
        return self.summary(cid)

    def _record(self, cid: str) -> dict:
        try:
            return self.records[cid]
        except KeyError as exc:
            raise KeyError(f"unknown conversation_id {cid!r}") from exc

    def history_id(self, cid: str) -> str:
        return self._record(cid)["reactor_history_id"]

    def update_history(self, cid: str, history_id: str) -> None:
        record = self._record(cid)
        record["reactor_history_id"], record["updated_at"] = history_id, _now()
        append_jsonl(self.path, {"kind": "conversation", "conversation_id": cid, "conversation": record})

    def append_answer(self, cid: str, answer: AnswerResponse) -> None:
        self._record(cid)
        now = _now()
        self.saved_answers[cid].append(answer)
        self.records[cid]["updated_at"] = now
        append_jsonl(self.path, {"kind": "answer", "conversation_id": cid, "at": now,
                                 "answer": answer.model_dump()})

    def summary(self, cid: str) -> Conversation:
        r = self._record(cid)
        return Conversation(conversation_id=cid, title=r["title"], created_at=r["created_at"],
                            updated_at=r["updated_at"], answer_count=len(self.saved_answers[cid]))

    def list(self) -> list[Conversation]:
        return sorted((self.summary(cid) for cid in self.records), key=lambda item: item.updated_at, reverse=True)

    def answers(self, cid: str) -> list[AnswerResponse]:
        self._record(cid)
        return list(self.saved_answers[cid])


def _response(result: QuestionResult, prediction: Prediction) -> AnswerResponse:
    return AnswerResponse(question_id=result.question_id, history_id=result.history_id, status=result.status,
                          abstain=prediction.abstain, answer=prediction.answer, citations=prediction.citations,
                          evidence_ids=result.sent_ids, guessed_type=result.guessed_type,
                          latency_s=result.latency_s, error=result.error)


class PipelineService:
    """Stateful Reactor facade; lock enforces the single-active-history agent limit."""

    def __init__(self, *, questions: list[dict], retrieval: Retrieval,
                 sessions: Mapping[str, Mapping[str, str]], answerer: Answerer | None = None,
                 state_path: str | Path = "conversations.jsonl", max_chars: int = 16_000):
        self.questions = {q["question_id"]: q for q in questions}
        self.retrieval, self.sessions = retrieval, sessions
        self.answerer, self.max_chars, self.policy = answerer or ReactorAnswerer(), max_chars, Policy()
        self.conversations, self.lock = ConversationStore(state_path), threading.Lock()

    @classmethod
    def from_files(cls, *, data_dir: str | Path, evidence_path: str | Path,
                   answerer: Answerer | None = None, state_path: str | Path = "conversations.jsonl",
                   max_chars: int = 16_000) -> "PipelineService":
        split = Split(data_dir)
        return cls(questions=split.questions, retrieval=FileRetrieval(evidence_path), sessions=split.sessions(),
                   answerer=answerer, state_path=state_path, max_chars=max_chars)

    def _row(self, question_id: str) -> Retrieved:
        try:
            question = self.questions[question_id]
        except KeyError as exc:
            raise KeyError(f"unknown question_id {question_id!r}") from exc
        return self.retrieval.retrieve([question])[0]

    def _answer(self, row: Retrieved, reactor_history_id: str) -> AnswerResponse:
        result = answer_one(row, self.answerer, self.max_chars, reactor_history_id)
        known = set(self.sessions.get(row.history_id, {})) or None
        return _response(result, to_prediction(result, self.policy, known))

    def create_conversation(self, title: str = "") -> Conversation:
        with self.lock:
            return self.conversations.create(self.answerer.start_history("conversation"), title)

    def answer_question(self, question_id: str) -> AnswerResponse:
        row = self._row(question_id)
        with self.lock:
            return self._answer(row, self.answerer.start_history(row.history_id))

    def answer_retrieved(self, row: Retrieved) -> AnswerResponse:
        with self.lock:
            return self._answer(row, self.answerer.start_history(row.history_id))

    def answer_in_conversation(self, cid: str, question_id: str) -> AnswerResponse:
        row = self._row(question_id)
        with self.lock:
            history_id = self.conversations.history_id(cid)
            if hasattr(self.answerer, "history_id"):
                self.answerer.history_id = history_id
            answer = self._answer(row, history_id)
            current = getattr(self.answerer, "history_id", None)
            if current:
                self.conversations.update_history(cid, current)
            self.conversations.append_answer(cid, answer)
            return answer


def create_app(service: PipelineService):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed; run `uv sync --extra api` first") from exc
    app = FastAPI(title="RxAI Memory Pipeline", version="0.1.0")

    def missing(call):
        try:
            return call()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "provider": "reactor", "questions": len(service.questions), "serialized": True}

    @app.post("/v1/dataset/answers/{question_id}", response_model=AnswerResponse)
    def answer_dataset(question_id: str): return missing(lambda: service.answer_question(question_id))

    @app.post("/v1/answers", response_model=AnswerResponse)
    def answer_retrieved(row: Retrieved): return service.answer_retrieved(row)

    @app.post("/v1/conversations", response_model=Conversation)
    def create_conversation(request: ConversationRequest): return service.create_conversation(request.title)

    @app.get("/v1/conversations", response_model=list[Conversation])
    def list_conversations(): return service.conversations.list()

    @app.get("/v1/conversations/{conversation_id}", response_model=Conversation)
    def get_conversation(conversation_id: str): return missing(lambda: service.conversations.summary(conversation_id))

    @app.get("/v1/conversations/{conversation_id}/answers", response_model=list[AnswerResponse])
    def list_answers(conversation_id: str): return missing(lambda: service.conversations.answers(conversation_id))

    @app.post("/v1/conversations/{conversation_id}/dataset/answers/{question_id}", response_model=AnswerResponse)
    def answer_conversation(conversation_id: str, question_id: str):
        return missing(lambda: service.answer_in_conversation(conversation_id, question_id))

    return app


def serve(*, data_dir: str | Path, evidence_path: str | Path, host: str = "127.0.0.1", port: int = 8000,
          state_path: str | Path = "conversations.jsonl", max_chars: int = 16_000,
          attempts: int = 3, timeout: float | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Uvicorn is not installed; run `uv sync --extra api` first") from exc
    app = create_app(PipelineService.from_files(data_dir=data_dir, evidence_path=evidence_path,
                     answerer=ReactorAnswerer(attempts=attempts, timeout=timeout), state_path=state_path,
                     max_chars=max_chars))
    uvicorn.run(app, host=host, port=port)
