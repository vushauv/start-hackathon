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
    return TestClient(create_app(State(runs, dry_run=True)))


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


# ---------------------------------------------------------------------------- reading-room API
from pipeline.models import Evidence, Retrieved  # noqa: E402

FIG_SESSIONS = [
    {"session_id": "s_x", "date": "2023/05/28 (Sun) 11:55", "turns": [
        {"turn_id": "t_1", "role": "user", "content": "Hi. I just adopted a cat—her name is Fig—and she rearranged my desk."},
        {"turn_id": "t_2", "role": "assistant", "content": "Of course, Carla! Fig sounds great."},
    ]},
    {"session_id": "s_y", "date": "2023/06/02 (Fri) 09:00", "turns": [
        {"turn_id": "t_3", "role": "user", "content": "Weather is nice today."},
    ]},
]


class StubRetrieval:
    def retrieve(self, questions):
        q = questions[0]
        return [Retrieved(question_id=q["question_id"], history_id=q["history_id"], question=q["question"],
                          question_date=q["question_date"], guessed_type="single-session-user",
                          evidence=[Evidence(id="s_x", date="2023/05/28 (Sun) 11:55", text="user: I just adopted a cat")],
                          candidates=["s_x", "s_y"])]


@pytest.fixture
def web(tmp_path, monkeypatch):
    d = tmp_path / "data" / "mini"
    write_jsonl(d / "questions.jsonl", QUESTIONS)
    write_jsonl(d / "histories.jsonl", [{"history_id": "h1", "sessions": FIG_SESSIONS}, HISTORIES[1]])
    monkeypatch.setenv("PIPELINE_DATA_ROOT", str(tmp_path / "data"))
    st = State(tmp_path / "runs", dry_run=True)
    st.run_jobs_inline = True
    st.local["mini"] = StubRetrieval()
    return TestClient(create_app(st)), st


def test_people_history_session(web):
    c, _ = web
    assert c.get("/api/meta").json()["dry_run"] is True
    p = c.get("/api/people", params={"split": "mini"}).json()["people"]
    h1 = next(x for x in p if x["history_id"] == "h1")
    assert h1["n_sessions"] == 2 and h1["first_date"].startswith("2023/05/28") and len(h1["questions"]) == 2
    hs = c.get("/api/history", params={"split": "mini", "history_id": "h1"}).json()["sessions"]
    assert [s["session_id"] for s in hs] == ["s_x", "s_y"] and hs[0]["preview"].startswith("Hi.")
    s = c.get("/api/session", params={"split": "mini", "history_id": "h1", "session_id": "s_y"}).json()
    assert s["prev"] == "s_x" and s["next"] is None and s["turns"][0]["turn_id"] == "t_3"
    assert c.get("/api/session", params={"split": "mini", "history_id": "h1", "session_id": "nope"}).status_code == 404
    assert c.get("/api/history", params={"split": "mini", "history_id": "zz"}).status_code == 404


def test_ask_job_returns_located_quote(web):
    c, _ = web
    r = c.post("/api/jobs", json={"split": "mini", "history_id": "h1", "question": "What is my cat's name?"})
    assert r.status_code == 202
    job = c.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["stage"] == "done", job
    res = job["result"]
    assert res["question_date"].startswith("2023/06/02")  # latest session of the history
    cite = res["citations"][0]
    assert cite["session_id"] == "s_x" and cite["label"] == "E1"
    q = cite["quotes"][0]
    assert q["turn_id"] == "t_1" and FIG_SESSIONS[0]["turns"][0]["content"][q["start"]:q["end"]] == q["text"]
    assert "Fig" in q["text"]


def test_job_errors(web):
    c, st = web
    assert c.post("/api/jobs", json={"split": "mini", "history_id": "h1", "question": " "}).status_code == 400
    assert c.get("/api/jobs/nope").status_code == 404
    del st.local["mini"]  # no retriever installed in the test env -> 503 reported in the job
    r = c.post("/api/jobs", json={"split": "mini", "history_id": "h1", "question": "x"})
    job = c.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["stage"] == "error" and "retriev" in job["error"]


@pytest.mark.parametrize("bad", ["../mini", "..", "mini/../..", "a/b", "%2e%2e"])
def test_split_name_cannot_escape(client, bad):
    assert client.get("/api/questions", params={"split": bad}).status_code == 400
    assert client.get("/api/people", params={"split": bad}).status_code == 400
    assert client.post("/api/predict", json={"split": bad}).status_code in (400, 422)
    assert client.post("/api/jobs", json={"split": bad, "history_id": "h1", "question": "x"}).status_code == 400
