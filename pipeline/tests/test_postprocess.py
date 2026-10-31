from pipeline.models import QuestionResult
from pipeline.postprocess import Policy, strip_labels, to_prediction


def result(**kw):
    base = dict(question_id="q1", history_id="h1", status="answered", answer_raw="June [E2].",
                cited_ids=["s_2"], sent_ids=["s_1", "s_2", "s_3"])
    return QuestionResult(**{**base, **kw})


def test_strip_labels():
    assert strip_labels("In June [E2].") == "In June."
    assert strip_labels("45 minutes [E1, E3] each way [ E2 ]") == "45 minutes each way"
    assert strip_labels("The answer is June [E2]") == "June"


def test_answered_keeps_agent_citations():
    p = to_prediction(result())
    assert (p.abstain, p.answer, p.citations) == (False, "June.", ["s_2"])


def test_answered_uncited_falls_back_to_top_sent():
    p = to_prediction(result(cited_ids=[], uncited=True), Policy(fallback_citations=2))
    assert p.citations == ["s_1", "s_2"]
    assert to_prediction(result(cited_ids=[]), Policy(fallback_citations=0)).citations == []


def test_abstained():
    p = to_prediction(result(status="abstained", answer_raw="INSUFFICIENT_EVIDENCE", cited_ids=[]))
    assert p.abstain and p.answer == "" and p.citations == ["s_1"]
    p = to_prediction(result(status="abstained", cited_ids=["s_3"]))
    assert p.citations == ["s_3"]


def test_no_answer_policy():
    r = result(status="no-answer", answer_raw="", cited_ids=[], error="TimedOut")
    assert to_prediction(r).abstain is True
    p = to_prediction(r, Policy(abstain_on_no_answer=False))
    assert (p.abstain, p.answer) == (False, "")


def test_unknown_session_ids_are_dropped():
    p = to_prediction(result(cited_ids=["s_9", "s_2", "s_2"]), known={"s_1", "s_2", "s_3"})
    assert p.citations == ["s_2"]
