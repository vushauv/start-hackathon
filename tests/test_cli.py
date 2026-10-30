import json
from pathlib import Path

import pytest

from memtest.cli import main
from memtest.pack import MAX_CHARS

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_offline_cli_writes_a_report(tmp_path, capsys):
    code = main([
        "offline",
        "--data-dir", str(FIXTURES),
        "--split", "dev",
        "--retriever", "bm25",
        "--limit", "7",
        "--runs-dir", str(tmp_path),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "evidence_f1" in out
    assert "knowledge-update" in out
    assert "single-session-assistant" in out
    reports = list(tmp_path.glob("*/report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["retriever"] == "bm25"
    assert report["split"] == "dev"
    assert report["limit"] == 7
    assert report["max_chars"] == MAX_CHARS
    assert report["metrics"]["all"]["n"] == 7
    details = [json.loads(line) for line in (reports[0].parent / "details.jsonl").read_text().splitlines()]
    by_id = {row["question_id"]: row for row in details}
    assert by_id["q_degree"]["ranked"][0] == "s_degree"
    assert by_id["q_degree"]["mrr"] == 1.0
    assert by_id["q_plugh"]["pred_abstain"] is True
    assert by_id["q_plugh"]["evidence_f1"] is None
    assert by_id["q_oil"]["sent"]
    for row in details:
        assert row["message_chars"] is not None
        assert row["message_chars"] <= MAX_CHARS


def test_unknown_retriever_and_bad_limit(tmp_path, capsys):
    assert main(["offline", "--data-dir", str(FIXTURES), "--split", "dev", "--retriever", "nope", "--runs-dir", str(tmp_path)]) == 2
    assert main(["offline", "--data-dir", str(FIXTURES), "--split", "dev", "--limit", "0", "--runs-dir", str(tmp_path)]) == 2


def test_official_score_wrapper(tmp_path, capsys):
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    gold.write_text(json.dumps({
        "question_id": "q_degree",
        "question_type": "single-session-user",
        "abstain": False,
        "answer": "marine biology",
        "aliases": [],
        "evidence": ["s_degree"],
    }) + "\n", encoding="utf-8")
    predictions.write_text(json.dumps({
        "question_id": "q_degree",
        "abstain": False,
        "answer": "You graduated with a degree in marine biology.",
        "citations": ["s_degree"],
    }) + "\n", encoding="utf-8")
    code = main(["score", "--gold", str(gold), "--predictions", str(predictions)])
    assert code == 0
    out = capsys.readouterr().out
    assert "single-session-user" in out
    assert "1.000" in out


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
