"""FastAPI adapter for the proven retrieval -> Reactor -> prediction pipeline.

The Reactor agent is stateful and accepts only one active request, so ``PipelineService`` deliberately
serializes requests. Retrieval stays offline: start the service with a prebuilt evidence JSONL, then use
``POST /v1/dataset/answers/{question_id}`` for dataset questions, or supply a fully retrieved bundle to
``POST /v1/answers`` from another retrieval system.
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
