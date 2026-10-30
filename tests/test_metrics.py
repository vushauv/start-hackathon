import importlib.util

import pytest

from memtest.kit import SCORE_PY
from memtest.metrics import (
    evidence_f1,
    format_table,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    score_question,
    summarize,
)


def _official_f1():
    spec = importlib.util.spec_from_file_location("rxai_score", SCORE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evidence_f1


def test_evidence_f1_matches_official_score():
    official = _official_f1()
    cases = [
        ([], ["s1"]),
        (["s1"], ["s1"]),
        (["s1"], ["s1", "s2"]),
        (["s1", "s9"], ["s1", "s2"]),
        (["s1", "s1"], ["s1"]),
        ([], []),
    ]
    for cited, gold in cases:
        assert evidence_f1(cited, gold) == pytest.approx(official(cited, gold))


def test_rank_metrics():
    ranked = ["s_weather", "s_degree"]
    gold = ["s_degree", "s_new"]
    assert recall_at_k(ranked, gold, 1) == pytest.approx(0.0)
    assert recall_at_k(ranked, gold, 2) == pytest.approx(0.5)
    assert precision_at_k(ranked, gold, 5) == pytest.approx(1 / 5)
    assert mean_reciprocal_rank(ranked, gold) == pytest.approx(0.5)
    assert evidence_f1(["s_degree"], gold) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, [], 1) is None


def test_summary_splits_abstention_from_answerable_retrieval():
    records = [
        score_question(
            question_id="q1",
            question_type="knowledge-update",
            gold_abstain=False,
            gold_evidence=["s_new", "s_old"],
            pred_abstain=False,
            ranked=["s_new", "s_old"],
            sent=["s_new"],
            top_score=1.5,
            message_chars=100,
            k_values=(1, 5),
        ),
        score_question(
            question_id="q2",
            question_type="knowledge-update",
            gold_abstain=True,
            gold_evidence=[],
            pred_abstain=True,
            ranked=[],
            sent=[],
            top_score=None,
            message_chars=80,
            k_values=(1, 5),
        ),
    ]
    rows = summarize(records, (1, 5))
    ku = rows["knowledge-update"]
    assert ku["n"] == 2
    assert ku["n_answerable"] == 1
    assert ku["abstention_n"] == 1
    assert ku["abstention_recall"] == pytest.approx(1.0)
    assert ku["false_abstention"] == pytest.approx(0.0)
    assert ku["recall@1"] == pytest.approx(0.5)
    assert ku["mrr"] == pytest.approx(1.0)
    assert ku["evidence_f1"] == pytest.approx(2 / 3)
    assert records[1]["evidence_f1"] is None
    table = format_table(rows, (1, 5))
    assert "knowledge-update" in table
    assert "evidence_f1" in table
    assert "\nall " in table or table.splitlines()[-1].startswith("all")
