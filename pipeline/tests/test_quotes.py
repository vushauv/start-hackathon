from pipeline.quotes import locate

FIG = {
    "session_id": "s_fig",
    "date": "2023/05/28 (Sun) 11:55",
    "turns": [
        {"turn_id": "t1", "role": "user", "content": "Hey, I’m trying to get my week in order. I just adopted a cat—her name is Fig—and she’s already rearranged my desk twice. Can you help me plan a few things?"},
        {"turn_id": "t2", "role": "assistant", "content": "Of course, Carla! Fig sounds like a handful in the best way. What do you need to plan?"},
        {"turn_id": "t3", "role": "user", "content": "Perfect, set the reminder for Thursday at 5pm. And maybe one for Saturday to clean my workspace before class."},
        {"turn_id": "t4", "role": "user", "content": "My son's school play is on 2023-06-10, so I need to shift a client call that Friday."},
    ],
}


def test_name_question_points_at_the_sentence_with_the_name():
    q = locate(FIG, "What is my cat's name?", "Fig")[0]
    c = FIG["turns"][0]["content"]
    assert (q.turn_id, q.role) == ("t1", "user")
    assert c[q.start:q.end] == q.text
    assert "her name is Fig" in q.text and "week in order" not in q.text


def test_date_question_prefers_the_turn_with_the_date():
    q = locate(FIG, "When is my son's school play?", "2023-06-10")[0]
    assert q.turn_id == "t4"
    assert FIG["turns"][3]["content"][q.start:q.end] == q.text


def test_excerpt_bonus_and_second_quote_threshold():
    qs = locate(FIG, "What is my cat's name?", "Fig", max_quotes=2)
    assert qs[0].turn_id == "t1"
    assert all(x.score >= 0.6 * qs[0].score for x in qs)


def test_fallback_first_user_turn_when_nothing_overlaps():
    q = locate(FIG, "zzz", "qqq")
    assert len(q) == 1 and q[0].turn_id == "t1" and q[0].text.endswith("order.")


def test_empty_session():
    assert locate({"session_id": "s", "turns": []}, "x", "y") == []
