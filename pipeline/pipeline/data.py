"""Reading a split: questions.jsonl, gold.jsonl, and only the histories that are needed."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SPLIT_FILES = ("questions.jsonl", "histories.jsonl")
_HID = re.compile(r'"history_id":\s*"([^"]+)"')


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def data_root() -> Path:
    """``$PIPELINE_DATA_ROOT``, else ``$RETRIEVER_DATA_ROOT`` (the retriever's variable), else ``<repo>/data``."""
    for var in ("PIPELINE_DATA_ROOT", "RETRIEVER_DATA_ROOT"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    return REPO / "data"


def split_dir(split: str, root: str | Path | None = None) -> Path:
    """The directory holding a split's files. Falls back to the repo root (where the README puts dev)."""
    base = Path(root).expanduser() if root else data_root()
    cands = [base / split] + ([REPO] if split == "dev" else [])  # the README puts dev's files in the repo root
    for cand in cands:
        if all((cand / f).exists() for f in SPLIT_FILES):
            return cand
    raise FileNotFoundError(
        f"split {split!r}: no questions.jsonl + histories.jsonl under {base / split}; "
        "pass --data DIR or set PIPELINE_DATA_ROOT (see README 'Data setup')"
    )


def load_histories(path: str | Path, wanted: set[str] | None = None) -> dict[str, dict]:
    """``{history_id: history}`` for the wanted ids only; other lines are skipped without parsing."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if wanted is not None:
                m = _HID.search(line[:200])
                if m and m.group(1) not in wanted:
                    continue
            h = json.loads(line)
            if wanted is None or h["history_id"] in wanted:
                out[h["history_id"]] = h
    return out


def session_dates(histories: dict[str, dict]) -> dict[str, dict[str, str]]:
    """``{history_id: {session_id: date}}``: what a citation must be, and the date to show for it."""
    return {hid: {s["session_id"]: s.get("date", "") for s in h["sessions"]} for hid, h in histories.items()}


class Split:
    """A split's questions, gold (if any) and lazily loaded histories."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.questions: list[dict] = read_jsonl(self.dir / "questions.jsonl")
        gold_path = self.dir / "gold.jsonl"
        self.gold: dict[str, dict] = (
            {g["question_id"]: g for g in iter_jsonl(gold_path)} if gold_path.exists() else {}
        )
        self._histories: dict[str, dict] | None = None

    @property
    def history_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for q in self.questions:
            seen.setdefault(q["history_id"], None)
        return list(seen)

    def histories(self) -> dict[str, dict]:
        if self._histories is None:
            self._histories = load_histories(self.dir / "histories.jsonl", set(self.history_ids))
        return self._histories

    def sessions(self) -> dict[str, dict[str, str]]:
        return session_dates(self.histories())
