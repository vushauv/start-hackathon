import pytest

from reactor_service.models import AskRequest, Evidence
from reactor_service.protocol import build_message, citations, final_answers, is_abstention

EV = [
    Evidence(id="s_1", date="2023/05/20 (Sat) 02:21", text="user: We booked the hall for May."),
    Evidence(id="s_2", date="2023/05/27 (Sat) 10:05", text="user: The wedding moved to June."),
]


def test_build_message_labels_and_ref():
    m = build_message("When is the wedding?", EV, ref="deadbeef", question_date="2023-06-01")
    assert "[ref:deadbeef]" in m
    assert "[E1] (id=s_1, date=2023/05/20 (Sat) 02:21)" in m
    assert "[E2] (id=s_2" in m
    assert "Question date: 2023-06-01" in m
    assert "# FINAL ANSWER [ref:deadbeef]:" in m
    assert "INSUFFICIENT_EVIDENCE" in m


def test_build_message_notes_block():
    m = build_message("When?", EV, ref="deadbeef", notes="[hint] E2 is the latest session")
    body, _, tail = m.partition("Notes:\n")
    assert "[E2]" in body and tail.startswith("[hint] E2 is the latest session")
    assert "# FINAL ANSWER" in tail  # the final-answer instruction comes after the notes
    assert "Notes:" not in build_message("When?", EV, ref="deadbeef", notes="  ")


def test_build_message_rejects_empty():
    with pytest.raises(ValueError):
        build_message("   ", EV, ref="deadbeef")


def test_final_answers_flexible_marker():
    reply = (
        "thinking...\n"
        "## final_answer [ref:AAAAAAAA] : old one\n"
        "# FINAL ANSWER [ref:deadbeef]: June [E2]\nmore text\n"
        "# FINAL ANSWER [ref:deadbeef]: June, same hall [E1, E2]"
    )
    out = final_answers(reply)
    assert out["aaaaaaaa"] == "old one"
    assert out["deadbeef"] == "June, same hall [E1, E2]"


def test_citations_known_and_unknown():
    cited, unknown = citations("June [E2] and [E1, E7] [E2]", EV)
    assert [(c.label, c.id) for c in cited] == [("E2", "s_2"), ("E1", "s_1")]
    assert unknown == ["E7"]


def test_abstention():
    assert is_abstention("INSUFFICIENT_EVIDENCE")
    assert is_abstention("insufficient evidence.")
    assert not is_abstention("June")


def test_ask_request_date_validation():
    AskRequest(question="q", question_date="2023-06-01")
    AskRequest(question="q", question_date="2023-06-01 14:30")
    with pytest.raises(ValueError):
        AskRequest(question="q", question_date="June 1st")
    with pytest.raises(ValueError):
        AskRequest(question=" ")
