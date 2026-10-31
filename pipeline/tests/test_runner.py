"""End to end against the fake Reactor: retrieval from a file, answering, resume, predictions, scoring."""

import json

from pipeline import Policy, run, write_predictions
from pipeline.answering import DryRunAnswerer, ReactorAnswerer
from pipeline.data import write_jsonl
from pipeline.retrieval import FileRetrieval
from pipeline.runner import reset_verification, verify_existing
from pipeline.scoring import score

QUESTIONS = [
    {"question_id": "q1", "history_id": "h1", "question": "Which month is the wedding?", "question_date": "2023/06/01 (Thu) 10:00"},
    {"question_id": "q2", "history_id": "h1", "question": "What is my hamster called?", "question_date": "2023/06/01 (Thu) 10:00"},
    {"question_id": "q3", "history_id": "h2", "question": "How long is my commute?", "question_date": "2023/06/02 (Fri) 10:00"},
]
EVIDENCE = [
    {"question_id": "q1", "history_id": "h1", "question": QUESTIONS[0]["question"], "question_date": QUESTIONS[0]["question_date"],
     "guessed_type": "knowledge-update", "hint": "[hint] most recent session = E2",
     "evidence": [{"id": "s_a", "date": "2023/05/20 (Sat) 02:21", "text": "user: hall booked for May"},
                  {"id": "s_b", "date": "2023/05/27 (Sat) 10:05", "text": "user: wedding moved to June"}],
     "candidates": ["s_a", "s_b", "s_c"]},
    {"question_id": "q2", "history_id": "h1", "question": QUESTIONS[1]["question"], "question_date": QUESTIONS[1]["question_date"],
     "guessed_type": "single-session-user", "hint": "",
     "evidence": [{"id": "s_c", "date": "2023/05/21 (Sun) 02:21", "text": "user: my cat is called Tom"}],
     "candidates": ["s_c"]},
    {"question_id": "q3", "history_id": "h2", "question": QUESTIONS[2]["question"], "question_date": QUESTIONS[2]["question_date"],
     "guessed_type": "single-session-user", "hint": "",
     "evidence": [{"id": "s_d", "date": "2023/05/22 (Mon) 02:21", "text": "user: 45 minutes each way"}],
     "candidates": ["s_d"]},
]
GOLD = [
    {"question_id": "q1", "question_type": "knowledge-update", "abstain": False, "answer": "June", "aliases": [], "evidence": ["s_b"]},
    {"question_id": "q2", "question_type": "single-session-user", "abstain": True, "answer": "no hamster", "aliases": [], "evidence": []},
    {"question_id": "q3", "question_type": "single-session-user", "abstain": False, "answer": "45 minutes", "aliases": [], "evidence": ["s_d"]},
]
SESSIONS = {"h1": {"s_a": "", "s_b": "", "s_c": ""}, "h2": {"s_d": ""}}


def agent(text: str):
    if "wedding" in text:
        return "June [E2]"
    if "hamster" in text:
        return "INSUFFICIENT_EVIDENCE"
    if "commute" in text:
        return "45 minutes each way"  # no citation: the fallback policy kicks in
    return None


def test_end_to_end(tmp_path, fake_reactor):
    srv = fake_reactor(agent)
    ev = tmp_path / "evidence.jsonl"
    write_jsonl(ev, EVIDENCE)
    results = tmp_path / "results.jsonl"
    answerer = ReactorAnswerer(settings=srv.settings(), attempts=2)
    out = run(QUESTIONS, FileRetrieval(ev), answerer, results, max_chars=16000, log=lambda s: None)

    assert [r.status for r in out] == ["answered", "abstained", "answered"]
    assert out[0].cited_ids == ["s_b"] and out[0].reactor_history_id == "fake-1"
    assert out[2].reactor_history_id == "fake-2" and out[2].uncited
    assert srv.histories == 2  # one Reactor history per dataset history
    assert "Notes:" in srv.messages[0] and "[hint] most recent session = E2" in srv.messages[0]
    assert "Question date: 2023/06/01 (Thu) 10:00" in srv.messages[0]

    # resume: nothing left to do, nothing sent
    again = run(QUESTIONS, FileRetrieval(ev), answerer, results, log=lambda s: None)
    assert again == [] and len(srv.messages) == 3

    preds = write_predictions(results, tmp_path / "predictions.jsonl", questions=QUESTIONS,
                              policy=Policy(fallback_citations=1, abstain_citations=1),
                              sessions=SESSIONS, log=lambda s: None)
    lines = [json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert lines == [
        {"question_id": "q1", "abstain": False, "answer": "June", "citations": ["s_b"]},
        {"question_id": "q2", "abstain": True, "answer": "", "citations": ["s_c"]},
        {"question_id": "q3", "abstain": False, "answer": "45 minutes each way", "citations": ["s_d"]},
    ]
    assert len(preds) == 3

    gold = tmp_path / "gold.jsonl"
    write_jsonl(gold, GOLD)
    rows = score(gold, tmp_path / "predictions.jsonl")
    assert rows["all"]["correct"] == 1.0 and rows["all"]["evidence_f1"] == 1.0
    assert rows["all"]["abstention_recall"] == 1.0 and rows["all"]["false_abstention"] == 0.0


def test_no_answer_is_recorded_and_missing_filled(tmp_path, fake_reactor):
    srv = fake_reactor(lambda text: None)  # the agent never writes a final answer -> timeout
    ev = tmp_path / "evidence.jsonl"
    write_jsonl(ev, EVIDENCE[2:])
    results = tmp_path / "results.jsonl"
    s = srv.settings().model_copy(update={"timeout": 0.5})
    out = run(QUESTIONS[2:], FileRetrieval(ev), ReactorAnswerer(settings=s, attempts=2), results, log=lambda s: None)
    assert out[0].status == "no-answer" and out[0].error.split(":")[0] in ("TimedOut", "Closed")
    assert out[0].attempts == 2 and len(srv.messages) == 2  # retried once, then gave up

    preds = write_predictions(results, tmp_path / "p.jsonl", questions=QUESTIONS,
                              policy=Policy(abstain_citations=1), log=lambda s: None)
    assert [p.abstain for p in preds] == [True, True, True]
    assert preds[2].citations == ["s_d"] and preds[0].citations == []


def test_dry_run(tmp_path):
    ev = tmp_path / "evidence.jsonl"
    write_jsonl(ev, EVIDENCE)
    out = run(QUESTIONS, FileRetrieval(ev), DryRunAnswerer(), tmp_path / "r.jsonl", log=lambda s: None)
    assert [r.cited_ids for r in out] == [["s_a"], ["s_c"], ["s_d"]]
    assert all(r.message_chars > 0 for r in out)


def test_stateless_answerer_can_run_in_parallel(tmp_path):
    ev = tmp_path / "evidence.jsonl"
    write_jsonl(ev, EVIDENCE)
    results = tmp_path / "parallel-results.jsonl"

    out = run(QUESTIONS, FileRetrieval(ev), DryRunAnswerer(), results, workers=3, log=lambda s: None)

    assert {r.question_id for r in out} == {"q1", "q2", "q3"}
    assert len([json.loads(line) for line in results.read_text().splitlines()]) == 3


def test_verifier_can_turn_an_unsupported_answer_into_an_abstention(tmp_path):
    class UnsupportedAnswerer(DryRunAnswerer):
        def verify(self, r, composed, answer):
            return "unsupported"

    ev = tmp_path / "evidence.jsonl"
    write_jsonl(ev, EVIDENCE[:1])

    out = run(QUESTIONS[:1], FileRetrieval(ev), UnsupportedAnswerer(), tmp_path / "results.jsonl", log=lambda s: None)

    assert out[0].status == "abstained"
    assert out[0].answer_raw == "INSUFFICIENT_EVIDENCE"
    assert out[0].verification == "unsupported"


def test_verify_existing_updates_saved_answer_without_reasking(tmp_path):
    class UnsupportedVerifier(DryRunAnswerer):
        def verify_candidate(self, r, composed, answer_text, cited_ids):
            return "unsupported"

    ev = tmp_path / "evidence.jsonl"
    results = tmp_path / "results.jsonl"
    write_jsonl(ev, EVIDENCE[:1])
    run(QUESTIONS[:1], FileRetrieval(ev), DryRunAnswerer(), results, log=lambda s: None)

    out = verify_existing(QUESTIONS[:1], FileRetrieval(ev), UnsupportedVerifier(), results, log=lambda s: None)

    assert out[0].status == "abstained"
    assert out[0].verification == "unsupported"

    assert reset_verification(results) == 1
    restored = [json.loads(line) for line in results.read_text().splitlines()]
    assert restored[0]["status"] == "answered"
    assert restored[0]["verification"] == "not-run"
