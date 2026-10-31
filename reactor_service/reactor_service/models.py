"""Pydantic models for Reactor inputs and outputs."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_DATE_RE = re.compile(
    r"^\d{4}[-/]\d{2}[-/]\d{2}"  # YYYY-MM-DD or YYYY/MM/DD
    r"(?:\s*\([A-Za-z]{3}\))?"  # optional (Sat)
    r"(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$"  # optional HH:MM[:SS]
)


class Evidence(BaseModel):
    """One passage (excerpt) sent to Reactor as ``[E<n>]``."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Your identifier, usually the session_id")
    text: str = Field(min_length=1)
    date: str | None = Field(default=None, description="Session date as it appears in the dataset")


class AskRequest(BaseModel):
    """A question plus the evidence to answer it from."""

    question: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    question_date: str | None = None
    history_id: str | None = None
    question_id: str | None = None
    notes: str | None = Field(default=None, description="Extra hints/rules sent after the evidence")

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v

    @field_validator("question_date")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not _DATE_RE.match(v):
            raise ValueError(
                "question_date: use YYYY-MM-DD, optionally followed by HH:MM, "
                "exactly as it appears in questions.jsonl"
            )
        return v


class Citation(BaseModel):
    label: str = Field(description="Evidence label, e.g. E1")
    id: str | None = Field(default=None, description="Your evidence id, None if unknown")


class History(BaseModel):
    """Response of ``POST /agents/{agent_id}/history``."""

    model_config = ConfigDict(extra="allow")

    history_id: str
    started_at: str | None = None
    running: bool | None = None
    restarted: bool | None = None


class RefusalInfo(BaseModel):
    """Payload of a server ``error`` frame."""

    code: str | None = None
    text: str | None = None
    forwarded: bool | None = None
    history_id: str | None = None
    current_history_id: str | None = None
    limit: int | None = None
    size: int | None = None


class TextReply(BaseModel):
    """Result of a free-text send (no ref / no final-answer parsing)."""

    status: Literal["reply"] = "reply"
    reply: str
    cycle: int
    latency_s: float
    retries: int = 0
    other_replies: list[str] = Field(default_factory=list)
    late_replies: list[str] = Field(default_factory=list)
    history_id: str | None = None


class Answer(BaseModel):
    """Result of :func:`reactor_service.ask`.

    Field names mirror the official kit's ``results.jsonl`` line so
    ``answer.model_dump()`` can be written straight to that file.
    """

    status: Literal["answered"] = "answered"
    answer: str
    abstained: bool
    citations: list[Citation] = Field(default_factory=list)
    unknown_citations: list[str] = Field(default_factory=list)
    uncited: bool = False
    reply: str
    cycle: int
    latency_s: float
    retries: int = 0
    other_replies: list[str] = Field(default_factory=list)
    late_replies: list[str] = Field(default_factory=list)
    ref: str
    question: str
    question_date: str | None = None
    question_id: str | None = None
    history_id: str | None = None

    @property
    def cited_ids(self) -> list[str]:
        """Evidence ids cited by the answer, deduplicated, in order."""
        seen: dict[str, None] = {}
        for c in self.citations:
            if c.id is not None:
                seen.setdefault(c.id, None)
        return list(seen)

    def to_jsonl(self) -> str:
        import json

        return json.dumps(self.model_dump(), ensure_ascii=True)


def _as_refusal(frame: dict[str, Any]) -> RefusalInfo:
    code = frame.get("code")
    return RefusalInfo(
        code=code if isinstance(code, str) else None,
        text=frame.get("text") if isinstance(frame.get("text"), str) else None,
        forwarded=frame.get("forwarded") if isinstance(frame.get("forwarded"), bool) else None,
        history_id=frame.get("history_id") if isinstance(frame.get("history_id"), str) else None,
        current_history_id=(
            frame.get("current_history_id")
            if isinstance(frame.get("current_history_id"), str)
            else None
        ),
        limit=frame.get("limit") if isinstance(frame.get("limit"), int) else None,
        size=frame.get("size") if isinstance(frame.get("size"), int) else None,
    )
