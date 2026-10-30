import json
from pathlib import Path

import pytest

from memtest.data import DataError, HistoryStore, index_jsonl, iter_jsonl, load_gold, load_questions, resolve_split

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_resolve_split_and_counts():
    paths = resolve_split(FIXTURES, "dev", need_gold=True)
    assert paths.root == FIXTURES / "dev"
    assert len(load_questions(paths.questions)) == 7
    gold = load_gold(paths.gold)
    assert gold["q_hamster"]["abstain"] is True
    assert gold["q_degree"]["evidence"] == ["s_degree"]


def test_resolve_split_missing(tmp_path):
    with pytest.raises(DataError):
        resolve_split(tmp_path, "dev", need_gold=True)


def test_iter_jsonl_is_a_stream():
    stream = iter_jsonl(FIXTURES / "dev" / "questions.jsonl")
    assert hasattr(stream, "__next__")
    first = next(stream)
    assert first["question_id"] == "q_degree"


def test_history_store_reads_by_id_and_evicts():
    store = HistoryStore(FIXTURES / "dev" / "histories.jsonl", cache_size=1)
    demo = store.get("h_demo")
    pet = store.get("h_pet")
    assert demo["sessions"][1]["session_id"] == "s_degree"
    assert pet["sessions"][0]["session_id"] == "s_cat"
    assert list(store._cache) == ["h_pet"]
    assert len(store) == 2


def test_index_uses_the_header_and_keeps_the_first_id(tmp_path):
    path = tmp_path / "histories.jsonl"
    path.write_bytes(
        b'{"history_id": "h_first", "sessions": [], "marker": 1}\n'
        b'{"history_id": "h_first", "sessions": [], "marker": 2}\n'
        b'{"history_id": "h_tail", "sessions": NOT_JSON_AFTER_THE_HEADER\n'
    )
    offsets = index_jsonl(path, "history_id")
    assert list(offsets) == ["h_first", "h_tail"]
    store = HistoryStore(path)
    assert store.get("h_first")["marker"] == 1


def test_bom_and_blank_lines(tmp_path):
    path = tmp_path / "histories.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"history_id": "h_bom", "sessions": []}\n\n')
    store = HistoryStore(path)
    assert store.get("h_bom")["history_id"] == "h_bom"
