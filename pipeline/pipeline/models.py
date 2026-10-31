"""Data passed between stages. Field names of ``Prediction`` are the submission format."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    """One passage sent to Reactor. ``id`` is the session_id, so a citation is a submission citation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    date: str = ""
    text: str


class Retrieved(BaseModel):
    """What retrieval produced for one question (one line of evidence_<split>.jsonl)."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    history_id: str
    question: str
    question_date: str = ""
    guessed_type: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    hint: str = ""
    candidates: list[str] = Field(default_factory=list, description="all candidate session ids, best first")


Status = Literal["answered", "abstained", "no-answer"]


class QuestionResult(BaseModel):
    """One line of results.jsonl: everything about one question, for predictions and for debugging."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    history_id: str
    status: Status
    answer_raw: str = ""
    cited_ids: list[str] = Field(default_factory=list, description="evidence ids the answer cites, in order")
    unknown_citations: list[str] = Field(default_factory=list)
    uncited: bool = False
    error: str = ""
    # context
    guessed_type: str = ""
    sent_ids: list[str] = Field(default_factory=list, description="evidence ids sent, best first")
    candidates: list[str] = Field(default_factory=list)
    message_chars: int = 0
    reactor_history_id: str = ""
    # reactor details
    reply: str = ""
    other_replies: list[str] = Field(default_factory=list)
    late_replies: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    retries: int = 0
    attempts: int = 1


class Prediction(BaseModel):
    """One line of predictions.jsonl, the submission format."""

    question_id: str
    abstain: bool
    answer: str
    citations: list[str]
