from memtest.kit import client
from memtest.pack import MAX_CHARS, client_passages, message_text, pack_evidence


QUESTION = "What ratio of tea tree oil to carrier oil should I use?"
WHEN = "2023/05/30 (Tue) 08:08"


def test_budget_constant_matches_the_vendored_client():
    assert MAX_CHARS == client().MAX_CHARS


def test_packer_keeps_roles_and_session_date_inside_the_budget():
    snippet = {
        "session_id": "s_degree",
        "date": "2023/04/15 (Sat) 18:02",
        "score": 2.0,
        "turns": [
            {"role": "user", "content": "I graduated with a degree in marine biology.", "score": 1.0},
            {"role": "assistant", "content": "Use a 1:10 ratio of tea tree oil to carrier oil.", "score": 2.0},
        ],
    }
    packed = pack_evidence(QUESTION, WHEN, [snippet])
    assert len(packed) == 1
    assert packed[0]["id"] == "s_degree"
    assert packed[0]["date"] == "2023/04/15 (Sat) 18:02"
    assert "user:" in packed[0]["text"]
    assert "assistant:" in packed[0]["text"]
    message = message_text(QUESTION, WHEN, client_passages(packed))
    assert len(message) <= MAX_CHARS
    assert "2023/04/15 (Sat) 18:02" in message
    assert "source: s_degree" in message


def test_packer_drops_a_long_user_turn_before_the_assistant_answer():
    snippet = {
        "session_id": "s1",
        "date": "2023/05/20 (Sat) 07:30",
        "score": 5.0,
        "turns": [
            {"role": "user", "content": "pad " * 8000, "score": 0.1},
            {"role": "assistant", "content": "Use a 1:10 ratio of tea tree oil to carrier oil.", "score": 5.0},
        ],
    }
    packed = pack_evidence(QUESTION, WHEN, [snippet])
    assert packed
    assert "assistant:" in packed[0]["text"]
    assert "tea tree oil" in packed[0]["text"]
    assert "pad pad" not in packed[0]["text"]
    assert len(message_text(QUESTION, WHEN, client_passages(packed))) <= MAX_CHARS


def test_packer_never_exceeds_the_message_limit():
    snippets = []
    for n in range(8):
        snippets.append({
            "session_id": f"s_{n}",
            "date": "2023/05/20 (Sat) 07:30",
            "score": float(8 - n),
            "turns": [{"role": "assistant", "content": ("evidence " * 400) + f" item {n}", "score": float(8 - n)}],
        })
    packed = pack_evidence(QUESTION, WHEN, snippets, max_chars=MAX_CHARS)
    assert packed
    assert packed[0]["id"] == "s_0"
    message = message_text(QUESTION, WHEN, client_passages(packed))
    assert len(message) <= MAX_CHARS
    # A later passage is allowed only when the whole client message still fits.
    assert all(p["id"].startswith("s_") for p in packed)
