from pathlib import Path

from memtest.data import HistoryStore, load_questions
from memtest.retriever import BM25Retriever

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _bundle():
    store = HistoryStore(FIXTURES / "dev" / "histories.jsonl")
    questions = {q["question_id"]: q for q in load_questions(FIXTURES / "dev" / "questions.jsonl")}
    return store, questions


def test_bm25_ranks_the_degree_session_first():
    store, questions = _bundle()
    snippets = BM25Retriever().retrieve(questions["q_degree"], store.get("h_demo"))
    assert snippets
    assert snippets[0]["session_id"] == "s_degree"
    assert snippets[0]["date"] == "2023/04/15 (Sat) 18:02"
    assert snippets[0]["score"] >= snippets[-1]["score"]
    assert "user:" in snippets[0]["text"]
    assert "assistant:" in snippets[0]["text"]


def test_bm25_keeps_assistant_turns_for_assistant_questions():
    store, questions = _bundle()
    snippets = BM25Retriever().retrieve(questions["q_oil"], store.get("h_demo"))
    assert snippets[0]["session_id"] == "s_degree"
    assert "assistant:" in snippets[0]["text"]
    assert "tea tree oil" in snippets[0]["text"]


def test_bm25_surfaces_both_5k_sessions():
    store, questions = _bundle()
    snippets = BM25Retriever().retrieve(questions["q_5k"], store.get("h_demo"))
    found = [s["session_id"] for s in snippets[:4]]
    assert "s_old" in found
    assert "s_new" in found
    assert found[0] in {"s_old", "s_new"}


def test_bm25_abstains_when_nothing_overlaps():
    store, questions = _bundle()
    snippets = BM25Retriever().retrieve(questions["q_plugh"], store.get("h_pet"))
    assert snippets == []


def test_unknown_retriever_name():
    import pytest
    from memtest.retriever import build_retriever

    with pytest.raises(KeyError):
        build_retriever("no-such")
