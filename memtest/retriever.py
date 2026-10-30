"""Pluggable retrieval.

A retriever maps a question and one history to a ranked list of snippets::

    {"session_id": "s_...", "date": "2023/05/20 (Sat) 02:21", "text": "user: ...\\nassistant: ...", "score": 1.2}

Higher scores come first. ``text`` includes the speaking role, and assistant
turns are indexed the same way as user turns. Gold evidence is session-level,
so each session appears at most once; its score is the best turn score in it.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

_TOKEN = re.compile(r"\d+(?:[:./]\d+)+|[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens. ``25:50`` stays one token."""

    return [match.group(0).lower() for match in _TOKEN.finditer(text or "")]


class Retriever(Protocol):
    """``retrieve(question, history)`` returns snippets best-first."""

    def retrieve(self, question: dict, history: dict) -> list[dict]:
        ...


def _finite(score: float) -> float:
    value = float(score)
    if math.isnan(value) or math.isinf(value) or value < 0:
        return 0.0
    return value


class BM25Retriever:
    """Okapi BM25 over turns, via ``rank_bm25``, collapsed to one snippet per session.

    The snippet keeps the highest-scoring turns and the turn on either side of
    each, so an assistant reply stays attached to the user turn it answers.
    Ties break toward the later session date, which is the right default for
    knowledge-update questions when the lexical score is the same.
    """

    name = "bm25"

    def __init__(self, top_sessions: int = 12, turns_per_session: int = 4, neighbor_turns: bool = True):
        if top_sessions < 1:
            raise ValueError("top_sessions must be at least 1")
        if turns_per_session < 1:
            raise ValueError("turns_per_session must be at least 1")
        self.top_sessions = top_sessions
        self.turns_per_session = turns_per_session
        self.neighbor_turns = neighbor_turns

    def retrieve(self, question: dict, history: dict) -> list[dict]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError('BM25 retrieval needs rank_bm25: pip install -r requirements.txt') from exc

        sessions = history.get("sessions") or []
        docs: list[tuple[int, int, list[str]]] = []
        for si, session in enumerate(sessions):
            if not isinstance(session, dict) or not isinstance(session.get("session_id"), str):
                continue
            for ti, turn in enumerate(session.get("turns") or []):
                if not isinstance(turn, dict):
                    continue
                docs.append((si, ti, tokenize(str(turn.get("content") or ""))))
        query = tokenize(str(question.get("question") or ""))
        if not docs or not query:
            return []

        corpus = [tokens if tokens else ["__empty__"] for _, _, tokens in docs]
        raw_scores = BM25Okapi(corpus).get_scores(query)
        by_session: dict[int, list[tuple[int, float]]] = {}
        best: dict[int, float] = {}
        for (si, ti, _), raw in zip(docs, raw_scores):
            score = _finite(raw)
            by_session.setdefault(si, []).append((ti, score))
            if score > best.get(si, -1.0):
                best[si] = score

        ranked_sessions = sorted(
            (si for si, score in best.items() if score > 0),
            key=lambda si: (best[si], str(sessions[si].get("date") or "")),
            reverse=True,
        )
        snippets = []
        for si in ranked_sessions[: self.top_sessions]:
            session = sessions[si]
            turn_scores = {ti: score for ti, score in by_session[si]}
            chosen = [
                ti
                for ti, score in sorted(by_session[si], key=lambda item: item[1], reverse=True)
                if score > 0
            ][: self.turns_per_session]
            keep = set(chosen)
            if self.neighbor_turns:
                n_turns = len(session.get("turns") or [])
                for ti in chosen:
                    if ti - 1 >= 0:
                        keep.add(ti - 1)
                    if ti + 1 < n_turns:
                        keep.add(ti + 1)
            cap = self.turns_per_session * (3 if self.neighbor_turns else 1)
            while len(keep) > cap:
                extras = [ti for ti in keep if ti not in chosen]
                if not extras:
                    break
                drop = min(extras, key=lambda ti: (turn_scores.get(ti, 0.0), -ti))
                keep.remove(drop)
            rendered = []
            for ti in sorted(keep):
                turn = session["turns"][ti]
                content = str(turn.get("content") or "").strip()
                if not content:
                    continue
                role = str(turn.get("role") or "user")
                rendered.append({
                    "role": role,
                    "content": content,
                    "score": turn_scores.get(ti, 0.0),
                })
            if not rendered:
                continue
            text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in rendered)
            snippets.append({
                "session_id": session["session_id"],
                "date": str(session.get("date") or ""),
                "text": text,
                "score": best[si],
                "turns": rendered,
            })
        return snippets


RETRIEVERS: dict[str, type] = {"bm25": BM25Retriever}


def register_retriever(name: str, factory: type) -> None:
    """Register a retriever class for ``--retriever``."""

    RETRIEVERS[name] = factory


def build_retriever(name: str, *, top_sessions: int = 12, turns_per_session: int = 4) -> Retriever:
    factory = RETRIEVERS.get(name)
    if factory is None:
        known = ", ".join(sorted(RETRIEVERS))
        raise KeyError(f"unknown retriever {name!r}; known: {known}")
    return factory(top_sessions=top_sessions, turns_per_session=turns_per_session)


def unique_session_ids(snippets: list[dict]) -> list[str]:
    """Session ids in rank order, first occurrence only."""

    ordered: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        sid = snippet.get("session_id")
        if isinstance(sid, str) and sid and sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    return ordered
