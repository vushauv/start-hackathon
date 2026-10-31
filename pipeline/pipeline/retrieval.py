"""Retrieval stage: questions -> ``Retrieved`` rows (evidence passages, hint, candidates).

Two implementations of the same ``Retrieval`` protocol:

* ``FileRetrieval``  reads an ``evidence_<split>.jsonl`` written earlier (by ``pipeline retrieve`` or
  ``retriever/build_evidence.py``). No GPU, no torch. This is how the answering stage runs on a laptop.
* ``LocalRetrieval`` wraps ``retriever/retriever.py`` (bge-small + BM25 + RRF + bge-reranker). Needs the
  ``retrieval`` extra and a GPU. Run it once per split on the server and keep the file.

To plug in another retriever (e.g. one backed by the Postgres chunks in ``db/``), implement ``retrieve()``
returning ``Retrieved`` rows: ``evidence[i].id`` must be a session_id of the question's history, because
the citations the agent makes are mapped straight to those ids.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from .data import REPO, iter_jsonl, write_jsonl
from .models import Retrieved

RETRIEVER_DIR = REPO / "retriever"


class Retrieval(Protocol):
    def retrieve(self, questions: list[dict]) -> list[Retrieved]:
        """One ``Retrieved`` per question, in the same order."""
        ...


class FileRetrieval:
    """Evidence rows from a JSONL file, keyed by question_id."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = {r["question_id"]: Retrieved.model_validate(r) for r in iter_jsonl(self.path)}

    def retrieve(self, questions: list[dict]) -> list[Retrieved]:
        missing = [q["question_id"] for q in questions if q["question_id"] not in self.rows]
        if missing:
            raise KeyError(f"{self.path}: no evidence for {len(missing)} question(s), e.g. {missing[:3]}; "
                           "run `pipeline retrieve` for this split first")
        return [self.rows[q["question_id"]] for q in questions]


class LocalRetrieval:
    """The team's GPU retriever (``retriever/retriever.py``), imported lazily so torch is optional."""

    def __init__(self, split_dir: str | Path, cache_dir: str | Path | None = None, *,
                 reranker: str | None = "base", rerank_text: str = "best2", fusion: str = "fused",
                 device: str = "cuda", log: Callable[[str], None] = print):
        if str(RETRIEVER_DIR) not in sys.path:
            sys.path.insert(0, str(RETRIEVER_DIR))
        import retriever as R  # noqa: PLC0415  (heavy: torch, sentence-transformers)

        self.R = R
        split_dir = Path(split_dir)
        cache_dir = Path(cache_dir) if cache_dir else RETRIEVER_DIR / "cache" / split_dir.name
        self.impl = R.Retriever(str(split_dir), str(cache_dir), reranker=reranker or None,
                                rerank_text_mode=rerank_text, fusion=fusion, device=device, log=log)

    def retrieve(self, questions: list[dict]) -> list[Retrieved]:
        return [Retrieved.model_validate(r) for r in self.impl.run(questions)]


def save_evidence(rows: Iterable[Retrieved], path: str | Path) -> None:
    write_jsonl(path, (r.model_dump() for r in rows))
