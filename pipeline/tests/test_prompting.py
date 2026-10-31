from pipeline.models import Evidence, Retrieved
from pipeline.prompting import ELLIPSIS, GENERAL_RULES, MIN_PASSAGE_CHARS, TYPE_RULES, compose, notes_for


def retrieved(n=3, size=6000, qtype="temporal-reasoning", hint="[hint] E1 -> E2: 7 days"):
    return Retrieved(question_id="q1", history_id="h1", question="How many days passed?",
                     question_date="2023/05/30 (Tue) 08:08", guessed_type=qtype, hint=hint,
                     evidence=[Evidence(id=f"s_{i}", date="2023/05/20 (Sat) 02:21", text=f"user: {'x' * size}")
                               for i in range(n)],
                     candidates=[f"s_{i}" for i in range(20)])


def test_notes_have_rules_type_rules_and_hint():
    n = notes_for(retrieved())
    assert GENERAL_RULES[0] in n
    assert TYPE_RULES["temporal-reasoning"][0] in n
    assert n.endswith("[hint] E1 -> E2: 7 days")
    assert TYPE_RULES["knowledge-update"][0] not in n

    knowledge_update_notes = notes_for(retrieved(qtype="knowledge-update"))
    assert "latest mention wins" in knowledge_update_notes


def test_unknown_type_only_general_rules():
    n = notes_for(retrieved(qtype="", hint=""))
    assert all(r in n for r in GENERAL_RULES) and "[hint]" not in n


def test_compose_fits_by_trimming_then_dropping():
    r = retrieved(n=3, size=6000)
    c = compose(r, 16000)
    assert c.chars <= 16000 and len(c.evidence) == 3
    assert sum(t.text.endswith(ELLIPSIS) for t in c.evidence) >= 1
    assert "Question [ref:" in c.message and "Notes:" in c.message and "[E3]" in c.message

    tight = compose(r, 2000)  # cannot fit three passages of 200 chars plus the notes: passages get dropped
    assert tight.chars <= 2000 and len(tight.evidence) < 3
    assert all(len(t.text) >= MIN_PASSAGE_CHARS for t in tight.evidence)


def test_compose_untouched_when_it_fits():
    r = retrieved(n=2, size=100)
    c = compose(r, 16000)
    assert [t.text for t in c.evidence] == [t.text for t in r.evidence]


def test_compose_sends_evidence_in_ascending_session_date_order():
    r = retrieved(n=3, size=100)
    r.evidence[0].date = "2023/05/27 (Sat) 10:05"
    r.evidence[1].date = "2023/05/20 (Sat) 02:21"
    r.evidence[2].date = "2023/05/21 (Sun) 02:21"

    c = compose(r, 16000)

    assert [e.id for e in c.evidence] == ["s_1", "s_2", "s_0"]
    assert c.notes.find("session dates matter as much as the text") >= 0


def test_retry_instruction_is_included_when_retrieval_provides_it():
    r = retrieved(hint="[retry] Review every supplied session before abstaining.")
    assert "Review every supplied session before abstaining." in notes_for(r)
