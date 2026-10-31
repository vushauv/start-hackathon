from pipeline.models import Evidence, Retrieved
from pipeline.retrieval import widen_evidence


def test_widen_evidence_keeps_selected_snippet_and_compacts_other_candidates():
    row = Retrieved(
        question_id="q1", history_id="h1", question="Question?", candidates=["s1", "s2"],
        evidence=[Evidence(id="s1", date="2023/05/01 (Mon) 09:00", text="selected relevant snippet")],
    )
    history = {"sessions": [
        {"session_id": "s1", "date": "2023/05/01 (Mon) 09:00", "turns": []},
        {"session_id": "s2", "date": "2023/05/02 (Tue) 09:00", "turns": [
            {"role": "user", "content": "start " + "a" * 4_000 + " end"},
        ]},
    ]}

    widened = widen_evidence(row, history, 2)

    assert widened.evidence[0].text == "selected relevant snippet"
    assert widened.evidence[1].text.startswith("user: start")
    assert widened.evidence[1].text.endswith("a" * 100 + " end")
    assert "facts that must be combined" in widened.hint
