"""JSONL loaders for histories, questions, and gold.

Histories are large (dev is about 249MB, dev-large about 744MB) and each line
is one user. ``HistoryStore`` keeps a byte-offset index keyed by ``history_id``
and reads a single line on demand, so a run over a slice of questions does not
hold every history in memory. The index itself stores offsets only: the id is
taken from the start of the line, which is where the release format puts
``history_id``.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


class DataError(Exception):
    """A split is missing, unreadable, or not in the release shape."""


_BOM = b"\xef\xbb\xbf"


def _strip_bom(line: bytes) -> bytes:
    return line[len(_BOM):] if line.startswith(_BOM) else line


def iter_jsonl(path: Path | str):
    """Yield objects from a JSONL file one line at a time."""

    path = Path(path)
    with path.open("rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(_strip_bom(raw))
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise DataError(f"{path}:{lineno}: expected a JSON object")
            yield obj


def index_jsonl(path: Path | str, id_field: str) -> dict[str, int]:
    """Map ``id_field`` to the byte offset of its line.

    The id is read from the first 1024 bytes of the line so the rest of a
    multi-megabyte history is not parsed during the scan. The first occurrence
    of an id wins.
    """

    path = Path(path)
    pattern = re.compile(rb'"' + re.escape(id_field.encode("utf-8")) + rb'"\s*:\s*"([^"\\]+)"')
    offsets: dict[str, int] = {}
    with path.open("rb") as fh:
        while True:
            offset = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            head = _strip_bom(raw)[:1024]
            match = pattern.search(head)
            if match is None:
                try:
                    obj = json.loads(_strip_bom(raw))
                except json.JSONDecodeError as exc:
                    raise DataError(f"{path}: no {id_field!r} near offset {offset} ({exc.msg})") from exc
                value = obj.get(id_field) if isinstance(obj, dict) else None
                if not isinstance(value, str):
                    raise DataError(f"{path}: line at offset {offset} has no string {id_field}")
            else:
                value = match.group(1).decode("utf-8")
            offsets.setdefault(value, offset)
    return offsets


def read_at(path: Path | str, offset: int) -> dict:
    """Parse the JSON object that starts at ``offset``."""

    path = Path(path)
    with path.open("rb") as fh:
        fh.seek(offset)
        raw = fh.readline()
    try:
        obj = json.loads(_strip_bom(raw))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: invalid JSON at offset {offset} ({exc.msg})") from exc
    if not isinstance(obj, dict):
        raise DataError(f"{path}: expected a JSON object at offset {offset}")
    return obj


class HistoryStore:
    """Random access to ``histories.jsonl`` with a small LRU of parsed records."""

    def __init__(self, path: Path | str, cache_size: int = 4):
        self.path = Path(path)
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.cache_size = cache_size
        self.offsets = index_jsonl(self.path, "history_id")
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, history_id: str) -> dict:
        cached = self._cache.get(history_id)
        if cached is not None:
            self._cache.move_to_end(history_id)
            return cached
        offset = self.offsets.get(history_id)
        if offset is None:
            raise DataError(f"history_id {history_id!r} is not in {self.path}")
        record = read_at(self.path, offset)
        if record.get("history_id") != history_id:
            raise DataError(
                f"offset for {history_id!r} in {self.path} points at {record.get('history_id')!r}"
            )
        self._cache[history_id] = record
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return record


@dataclass(frozen=True)
class SplitPaths:
    root: Path
    histories: Path
    questions: Path
    gold: Path | None


def resolve_split(data_dir: Path | str, split: str, *, need_gold: bool) -> SplitPaths:
    """Find ``{data_dir}/{split}/`` or, if that is absent, ``data_dir`` itself.

    Release archives extract as ``dev/histories.jsonl`` and the rest, so the
    usual layout is ``data/dev/``.
    """

    data_dir = Path(data_dir)
    candidates = [data_dir / split, data_dir]
    seen: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.append(candidate)
        questions = candidate / "questions.jsonl"
        histories = candidate / "histories.jsonl"
        if not (questions.is_file() and histories.is_file()):
            continue
        gold = candidate / "gold.jsonl"
        if need_gold and not gold.is_file():
            raise DataError(
                f"{candidate} has questions and histories but no gold.jsonl. "
                "Offline metrics need a dev split; test gold is not released."
            )
        return SplitPaths(
            root=candidate,
            histories=histories,
            questions=questions,
            gold=gold if gold.is_file() else None,
        )
    looked = ", ".join(str(c) for c in seen)
    raise DataError(
        f"no questions.jsonl and histories.jsonl for split {split!r} under {data_dir} "
        f"(looked in {looked}). Download the data-v1 release into data/ as the README describes."
    )


def load_questions(path: Path | str) -> list[dict]:
    questions = list(iter_jsonl(path))
    for n, question in enumerate(questions, 1):
        for field in ("question_id", "history_id", "question", "question_date"):
            if not isinstance(question.get(field), str) or not question[field]:
                raise DataError(f"{path}: question {n} is missing string field {field}")
    return questions


def load_gold(path: Path | str) -> dict[str, dict]:
    """Gold rows keyed by ``question_id``. Later duplicates replace earlier ones."""

    gold: dict[str, dict] = {}
    for n, row in enumerate(iter_jsonl(path), 1):
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise DataError(f"{path}: gold row {n} is missing question_id")
        if not isinstance(row.get("question_type"), str):
            raise DataError(f"{path}: gold row {qid} is missing question_type")
        if not isinstance(row.get("abstain"), bool):
            raise DataError(f"{path}: gold row {qid} is missing boolean abstain")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise DataError(f"{path}: gold row {qid} evidence must be a list of session ids")
        gold[qid] = row
    return gold


def select_questions(questions: list[dict], limit: int | None) -> list[dict]:
    """The first ``limit`` questions in file order, or every question."""

    if limit is None:
        return list(questions)
    if limit < 1:
        raise DataError("--limit must be a positive integer")
    return list(questions[:limit])
