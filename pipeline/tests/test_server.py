"""The web runner's API, in dry-run mode (no Reactor), on a temporary split."""

import pytest
from fastapi.testclient import TestClient

from pipeline.data import write_jsonl
from pipeline.server import State, create_app
from tests.test_runner import EVIDENCE, GOLD, QUESTIONS

HISTORIES = [
    {"history_id": "h1", "sessions": [{"session_id": s, "date": "2023/05/20 (Sat) 02:21", "turns": []} for s in ("s_a", "s_b", "s_c")]},
    {"history_id": "h2", "sessions": [{"session_id": "s_d", "date": "2023/05/22 (Mon) 02:21", "turns": []}]},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    d = tmp_path / "data" / "mini"
    write_jsonl(d / "questions.jsonl", QUESTIONS)
    write_jsonl(d / "histories.jsonl", HISTORIES)
    write_jsonl(d / "gold.jsonl", GOLD)
    monkeypatch.setenv("PIPELINE_DATA_ROOT", str(tmp_path / "data"))
    runs = tmp_path / "runs"
    write_jsonl(runs / "mini" / "evidence.jsonl", EVIDENCE)
    return TestClient(create_app(State(runs, dry_run=True), allowed_hosts=["testserver", "127.0.0.1", "localhost"]))


def test_splits_and_questions(client):
    s = client.get("/api/splits").json()
    assert s["splits"] == ["mini"] and s["dry_run"] is True
    q = client.get("/api/questions", params={"split": "mini"}).json()
    assert len(q["questions"]) == 3 and q["has_evidence"] and q["has_gold"]
    assert q["questions"][0]["gold"]["answer"] == "June" and q["questions"][0]["result"] is None


def test_question_ask_and_predict(client):
    d = client.get("/api/question", params={"split": "mini", "question_id": "q1"}).json()
    assert d["composed"]["evidence_ids"] == ["s_a", "s_b"] and "Notes:" in d["composed"]["message"]
    assert d["result"] is None

    out = client.post("/api/ask", json={"split": "mini", "question_id": "q1"}).json()
    assert out["result"]["status"] == "answered" and out["prediction"]["citations"] == ["s_a"]
    assert out["result"]["reactor_history_id"] == "dry-1"
    out2 = client.post("/api/ask", json={"split": "mini", "question_id": "q2"}).json()
    assert out2["result"]["reactor_history_id"] == "dry-1"  # same dataset history: Reactor history kept
    out3 = client.post("/api/ask", json={"split": "mini", "question_id": "q3"}).json()
    assert out3["result"]["reactor_history_id"] == "dry-2"  # other dataset history: fresh one

    d = client.get("/api/question", params={"split": "mini", "question_id": "q1"}).json()
    assert d["result"]["status"] == "answered" and d["prediction"]["abstain"] is False
    assert client.get("/api/questions", params={"split": "mini"}).json()["questions"][0]["result"]["status"] == "answered"

    p = client.post("/api/predict", json={"split": "mini"}).json()
    assert p["n"] == 3 and p["score"]["all"]["n"] == 3
    assert p["score"]["all"]["evidence_f1"] == 0.5  # dry run cites E1: right for q3, wrong for q1


def test_errors(client):
    assert client.get("/api/question", params={"split": "mini", "question_id": "nope"}).status_code == 404
    assert client.get("/api/questions", params={"split": "missing"}).status_code == 404
    r = client.post("/api/ask-custom", json={"split": "mini", "history_id": "h1", "question": "x"})
    assert r.status_code == 503 and "retrieval" in r.json()["detail"]
    assert client.get("/").status_code == 200 and "Pipeline runner" in client.get("/").text


def test_split_name_validation_and_host_guard(client):
    assert client.get("/api/questions", params={"split": "../etc"}).status_code == 400
    assert client.post("/api/predict", json={"split": "a/b"}).status_code == 400
    assert client.get("/api/splits", headers={"Host": "evil.example"}).status_code == 400
