import json
from pathlib import Path

from memtest.online import run_online

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SECRET = "rxk_test_secret_do_not_leak"


def _clear_creds(monkeypatch):
    for name in ("REACTOR_ACCESS_KEY", "REACTOR_BASE_URL", "REACTOR_AGENT_ID"):
        monkeypatch.delenv(name, raising=False)


def test_skips_cleanly_when_credentials_are_missing(tmp_path, monkeypatch, capsys):
    _clear_creds(monkeypatch)
    calls = []
    code = run_online(
        data_dir=Path("/no/such/data"),
        split="dev",
        retriever_name="bm25",
        limit=1,
        top_sessions=4,
        turns_per_session=2,
        max_chars=16000,
        runs_dir=tmp_path,
        results_path=None,
        base_url="https://example.test",
        agent_id="agt_test",
        abstain_on="agent,no-answer",
        invoke=lambda extra: calls.append(extra),
    )
    assert code == 0
    assert calls == []
    err = capsys.readouterr().err
    assert "REACTOR_ACCESS_KEY" in err
    assert "Skipping" in err
    assert SECRET not in err
    assert list(tmp_path.iterdir()) == []


def test_online_packs_evidence_resumes_and_writes_predictions(tmp_path, monkeypatch, capsys):
    _clear_creds(monkeypatch)
    monkeypatch.setenv("REACTOR_ACCESS_KEY", SECRET)
    calls = []
    evidence = []

    def invoke(extra):
        calls.append(list(extra))
        assert SECRET not in extra
        if "--new-history" in extra:
            return {"status": "history", "history_id": "reactor-h"}
        if "--evidence" in extra:
            path = Path(extra[extra.index("--evidence") + 1])
            evidence.append(json.loads(path.read_text(encoding="utf-8")))
        qid = extra[extra.index("--question-id") + 1]
        return {
            "status": "answered",
            "question_id": qid,
            "answer": "marine biology",
            "abstained": False,
            "citations": [{"label": "E1", "id": "s_degree"}],
        }

    code = run_online(
        data_dir=FIXTURES,
        split="dev",
        retriever_name="bm25",
        limit=2,
        top_sessions=4,
        turns_per_session=2,
        max_chars=16000,
        runs_dir=tmp_path,
        results_path=None,
        base_url="https://example.test",
        agent_id="agt_test",
        abstain_on="agent,no-answer",
        invoke=invoke,
    )
    assert code == 0
    new_history = [c for c in calls if "--new-history" in c]
    asked = [c[c.index("--question-id") + 1] for c in calls if "--question-id" in c]
    assert len(new_history) == 1
    assert asked == ["q_degree", "q_oil"]
    assert evidence
    assert evidence[0][0]["date"]
    assert any("assistant:" in p["text"] for p in evidence[0])
    blob = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file())
    assert SECRET not in blob
    predictions = list(tmp_path.rglob("predictions.jsonl"))
    assert len(predictions) == 1
    rows = [json.loads(line) for line in predictions[0].read_text(encoding="utf-8").splitlines()]
    assert {row["question_id"] for row in rows} == {"q_degree", "q_oil"}
    assert rows[0]["citations"] == ["s_degree"]
    assert rows[0]["abstain"] is False

    # Resume: q_degree and q_oil are already in results, so a second pass asks nothing new.
    calls.clear()
    results = next(tmp_path.rglob("results.jsonl"))
    code = run_online(
        data_dir=FIXTURES,
        split="dev",
        retriever_name="bm25",
        limit=2,
        top_sessions=4,
        turns_per_session=2,
        max_chars=16000,
        runs_dir=tmp_path,
        results_path=results,
        base_url="https://example.test",
        agent_id="agt_test",
        abstain_on="agent,no-answer",
        invoke=invoke,
    )
    assert code == 0
    assert calls == []
    err = capsys.readouterr().err
    assert "already in" in err
