"""Pack ranked snippets into the Reactor client's 16,000-character message.

The client builds one message from the question, the question date, the
passages, and its citation instructions, then refuses anything over
``--max-chars`` (default 16,000) before it sends. This packer uses that same
``build_message`` so the file we hand to ``--evidence`` fits, with each
passage's session id and session date preserved. Assistant turns stay in the
text until a lower-scoring turn has to be dropped to make the budget.
"""

from __future__ import annotations

from memtest.kit import client

MAX_CHARS = 16000
_REF = "0123abcd"  # same width as secrets.token_hex(4); length, not value, matters


def render_turns(turns: list[dict]) -> str:
    lines = []
    for turn in turns:
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        role = str(turn.get("role") or "user")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parsed_date(question_date: str | None) -> str | None:
    if not question_date:
        return None
    return client().parse_question_date(question_date)


def message_text(question: str, question_date: str | None, passages: list[dict]) -> str:
    """The exact string ``client.py`` would send for this question and evidence."""

    return client().build_message(question, passages, _REF, _parsed_date(question_date))


def message_length(question: str, question_date: str | None, passages: list[dict]) -> int:
    return len(message_text(question, question_date, passages))


def _passage(snippet: dict, text: str) -> dict:
    passage = {"id": snippet["session_id"], "text": text, "session_id": snippet["session_id"]}
    date = snippet.get("date")
    if isinstance(date, str) and date.strip():
        passage["date"] = date.strip()
    score = snippet.get("score")
    if isinstance(score, (int, float)):
        passage["score"] = float(score)
    return passage


def _fits(question: str, question_date: str | None, passages: list[dict], max_chars: int) -> bool:
    try:
        return message_length(question, question_date, passages) <= max_chars
    except ValueError:
        return False


def _truncate_turn(turn: dict, question: str, question_date: str | None, accepted: list[dict], snippet: dict, max_chars: int) -> str | None:
    role = str(turn.get("role") or "user")
    content = str(turn.get("content") or "").strip()
    if not content:
        return None
    best = None
    lo, hi = 1, len(content)
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = content[:mid].rstrip()
        if not chunk:
            lo = mid + 1
            continue
        text = f"{role}: {chunk}"
        if mid < len(content):
            text += "…"
        if _fits(question, question_date, accepted + [_passage(snippet, text)], max_chars):
            best = text
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _fit_snippet(question: str, question_date: str | None, accepted: list[dict], snippet: dict, max_chars: int) -> dict | None:
    turns = snippet.get("turns")
    if isinstance(turns, list) and turns:
        working = [dict(turn) for turn in turns if str(turn.get("content") or "").strip()]
    else:
        text = str(snippet.get("text") or "").strip()
        working = [{"role": "user", "content": text, "score": snippet.get("score") or 0.0}] if text else []
    if not working:
        return None

    def attempt(current: list[dict]) -> dict | None:
        text = render_turns(current)
        if not text:
            return None
        passage = _passage(snippet, text)
        if _fits(question, question_date, accepted + [passage], max_chars):
            return passage
        return None

    fitted = attempt(working)
    if fitted is not None:
        return fitted

    # Drop the least useful turn first. On a score tie, drop a non-assistant
    # turn before an assistant turn, and a later turn before an earlier one.
    while len(working) > 1:
        def drop_key(item: tuple[int, dict]) -> tuple:
            index, turn = item
            score = float(turn.get("score") or 0.0)
            assistant_last = 1 if turn.get("role") == "assistant" else 0
            return (score, assistant_last, -index)

        index, _turn = min(enumerate(working), key=drop_key)
        working.pop(index)
        fitted = attempt(working)
        if fitted is not None:
            return fitted

    truncated = _truncate_turn(working[0], question, question_date, accepted, snippet, max_chars)
    if truncated is None:
        return None
    return _passage(snippet, truncated)


def pack_evidence(question: str, question_date: str | None, snippets: list[dict], max_chars: int = MAX_CHARS) -> list[dict]:
    """Greedily add ranked snippets until the client message would exceed ``max_chars``.

    Each returned passage has ``id`` (the session id), ``date`` when the snippet
    has one, and role-labelled ``text``. Extra ``session_id`` and ``score`` keys
    are for metrics; ``client.py`` ignores them.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    accepted: list[dict] = []
    for snippet in snippets:
        if not isinstance(snippet, dict) or not isinstance(snippet.get("session_id"), str):
            continue
        fitted = _fit_snippet(question, question_date, accepted, snippet, max_chars)
        if fitted is not None:
            accepted.append(fitted)
    return accepted


def client_passages(packed: list[dict]) -> list[dict]:
    """The JSON list ``client.py --evidence`` expects."""

    passages = []
    for passage in packed:
        item = {"id": passage["id"], "text": passage["text"]}
        if passage.get("date"):
            item["date"] = passage["date"]
        passages.append(item)
    return passages
