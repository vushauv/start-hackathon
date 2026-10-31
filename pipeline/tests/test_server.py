from pipeline.answering import DryRunAnswerer
from pipeline.models import Evidence, Retrieved
from pipeline.retrieval import InMemoryRetrieval
from pipeline.server import PipelineService


QUESTION = {"question_id": "q1", "history_id": "h1", "question": "What is my pet called?"}
ROW = Retrieved(
    question_id="q1", history_id="h1", question=QUESTION["question"],
    evidence=[Evidence(id="s1", date="2023/05/01", text="user: My pet is Tofu.")],
)


def service(tmp_path):
    return PipelineService(questions=[QUESTION], retrieval=InMemoryRetrieval([ROW]),
                           sessions={"h1": {"s1": "2023/05/01"}}, answerer=DryRunAnswerer(),
                           state_path=tmp_path / "conversations.jsonl")


def test_dataset_question_runs_the_pipeline_and_returns_prediction(tmp_path):
    answer = service(tmp_path).answer_question("q1")
    assert answer.question_id == "q1"
    assert answer.status == "answered"
    assert answer.abstain is False
    assert answer.citations == ["s1"]
    assert answer.evidence_ids == ["s1"]


def test_direct_retrieved_bundle_uses_the_same_pipeline(tmp_path):
    answer = service(tmp_path).answer_retrieved(ROW)
    assert answer.answer == "(dry run)"


def test_unknown_question_is_rejected(tmp_path):
    try:
        service(tmp_path).answer_question("missing")
    except KeyError as exc:
        assert "unknown question_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_conversation_is_resumable_and_answers_survive_a_restart(tmp_path):
    first = service(tmp_path)
    conversation = first.create_conversation("demo")
    answer = first.answer_in_conversation(conversation.conversation_id, "q1")
    assert answer.citations == ["s1"]
    assert first.conversations.summary(conversation.conversation_id).answer_count == 1

    restarted = service(tmp_path)
    restored = restarted.conversations.summary(conversation.conversation_id)
    assert restored.title == "demo" and restored.answer_count == 1
    assert restarted.conversations.answers(conversation.conversation_id)[0] == answer
