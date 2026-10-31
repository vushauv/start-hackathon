"""Find the exact place in a session that an answer rests on.

Reactor cites whole sessions (``[E1]`` -> session id). The UI wants to open that session scrolled to the
turn, and highlight the sentence. ``locate`` scores each turn of the cited session against the question and
the answer and returns the best sentence(s) as character spans inside ``turn["content"]``.

Pure Python and deterministic: no torch, no network, so it runs in the web server and in tests.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

_TOK = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUM = re.compile(r"\d")

STOP = frozenset("""
a an and are as at be been but by can did do does for from had has have how i if in into is it its just me
my of on or our so than that the their them then there these they this to up was we were what when where
which who whom why will with would you your yours about after again all also am any because before being
both could each get got her him his not now only other out over she some such too very
""".split())


@dataclass
class Quote:
    session_id: str
    turn_id: str
    turn_index: int
    role: str
    start: int  # character span of the quoted sentence inside turn["content"]
    end: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def _weights(question: str, answer: str) -> dict[str, float]:
    w: dict[str, float] = {}
    for text, base in ((question, 1.0), (answer, 2.0)):
        for t in tokens(text):
            if t in STOP:
                continue
            f = 3.0 if _NUM.search(t) else base
            w[t] = max(w.get(t, 0.0), f)
    return w


def _best_sentence(content: str, w: dict[str, float]) -> tuple[int, int, float]:
    """Span of the sentence with the highest weight of distinct query tokens (first one wins ties)."""
    best = (0, 0, -1.0)
    pos = 0
    spans: list[tuple[int, int]] = []
    for m in _SENT.finditer(content):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(content)))
    for a, b in spans:
        if b <= a:
            continue
        sc = sum(w.get(t, 0.0) for t in set(tokens(content[a:b])))
        if sc > best[2]:
            best = (a, b, sc)
    if best[2] < 0:  # empty content
        return 0, len(content), 0.0
    return best


def locate(session: dict, question: str, answer: str, excerpt: str = "", max_quotes: int = 2) -> list[Quote]:
    """Best sentence(s) of ``session`` for this question/answer, most relevant first.

    ``excerpt`` is the text of this session that was sent to the model; a turn that appears in it gets a
    small bonus because it is what the retriever ranked highest.
    """
    w = _weights(question, answer)
    sid = session.get("session_id", "")
    scored: list[tuple[float, int, dict]] = []
    for i, t in enumerate(session.get("turns", [])):
        content = t.get("content") or ""
        if not content.strip():
            continue
        toks = tokens(content)
        sc = sum(w.get(x, 0.0) for x in set(toks)) / math.sqrt(max(len(toks), 1))
        sc *= 1.0 if t.get("role") == "user" else 0.6
        if excerpt and content.strip()[:80] in excerpt:
            sc += 0.25
        scored.append((sc, i, t))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[0][0]

    def quote(sc: float, i: int, t: dict, *, first_sentence: bool = False) -> Quote:
        content = t["content"]
        if first_sentence:
            a, b = 0, (_SENT.search(content).start() if _SENT.search(content) else len(content))
        else:
            a, b, _ = _best_sentence(content, w)
        # tidy leading whitespace so the highlight starts on a visible character
        while a < b and content[a].isspace():
            a += 1
        return Quote(sid, t.get("turn_id", ""), i, t.get("role", ""), a, b, content[a:b], round(sc, 3))

    if top <= 0.25 and not any(sc > 0.25 for sc, _, _ in scored):  # nothing overlaps: first user turn
        for sc, i, t in sorted(scored, key=lambda x: x[1]):
            if t.get("role") == "user":
                return [quote(0.0, i, t, first_sentence=True)]
        sc, i, t = sorted(scored, key=lambda x: x[1])[0]
        return [quote(0.0, i, t, first_sentence=True)]

    out = [quote(*scored[0])]
    for sc, i, t in scored[1:]:
        if len(out) >= max_quotes:
            break
        if sc >= 0.6 * top and sc > 0.25:
            out.append(quote(sc, i, t))
    return out
