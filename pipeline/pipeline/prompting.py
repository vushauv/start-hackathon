"""Prompting stage: what goes to Reactor besides question and evidence, and making it fit the limit.

The message template itself lives in ``reactor_service.protocol.build_message``. This module decides the
*notes* block (answer-style rules + per-type rules + the retriever's date hints) and shrinks evidence until
the whole message is under ``max_chars``.
"""

from __future__ import annotations

from dataclasses import dataclass

import reactor_service as reactor

from .models import Evidence, Retrieved

# Rules for every question. The judge marks hedging, several candidates or negation as wrong, so one
# short definite answer is the target; the user's own facts, not other people's (the test split's near
# misses are mostly "same fact about a friend / colleague").
GENERAL_RULES = [
    "Give exactly one definite answer in a short phrase (a name, number, date, duration or item). "
    "No hedging, no alternatives, no 'possibly'.",
    "The evidence is from the user's own past chats. 'I'/'my' in the question is the user. Facts about "
    "other people (friends, family, colleagues) do not answer a question about the user.",
    "If the evidence talks about the topic but does not state the asked fact for the user, answer "
    "INSUFFICIENT_EVIDENCE rather than guessing.",
    "Cite every [E#] passage the answer relies on, and no others.",
]

# Extra rules by guessed question type (the retriever's rule-based guess; wrong guesses are harmless).
TYPE_RULES = {
    "knowledge-update": [
        "The question asks for the current value. When passages disagree, the one from the most recent "
        "session date is the answer; earlier values are outdated.",
    ],
    "temporal-reasoning": [
        "Use the session dates given with each passage and in the hints. Day counts in the hints are "
        "computed exactly; prefer them to your own arithmetic. A date mentioned in the text as 'today', "
        "'yesterday' or 'last week' is relative to that passage's session date.",
    ],
    "single-session-assistant": [
        "The question is about what the assistant said or recommended. Take the answer from the "
        "'assistant:' turns, not from options the user mentioned.",
    ],
    "multi-session": [
        "The answer may combine facts from several passages (e.g. a count or a sum). Count only items "
        "that belong to the user and are explicitly stated; do not double count repeated mentions.",
    ],
}

MIN_PASSAGE_CHARS = 200
ELLIPSIS = " [...]"


def notes_for(r: Retrieved, *, general: list[str] | None = None, by_type: dict[str, list[str]] | None = None) -> str:
    """The notes block for one question: rules, type rules, and the retriever's hints."""
    rules = list(GENERAL_RULES if general is None else general)
    rules += (TYPE_RULES if by_type is None else by_type).get(r.guessed_type, [])
    lines = [f"- {x}" for x in rules]
    if r.hint.strip():
        lines.append("")
        lines.append(r.hint.strip())
    return "\n".join(lines)


@dataclass
class Composed:
    evidence: list[Evidence]
    notes: str
    message: str  # the exact text (with a dummy ref) to check sizes

    @property
    def chars(self) -> int:
        return len(self.message)


def _message(r: Retrieved, evidence: list[Evidence], notes: str) -> str:
    return reactor.build_message(
        r.question, [e.model_dump() for e in evidence],
        question_date=r.question_date or None, notes=notes,
    )


def compose(r: Retrieved, max_chars: int, *, notes: str | None = None) -> Composed:
    """Evidence shrunk so the message fits ``max_chars``: trim the longest passage, drop the last one if needed."""
    notes = notes_for(r) if notes is None else notes
    evidence = [e.model_copy() for e in r.evidence if e.text.strip()]
    while True:
        msg = _message(r, evidence, notes)
        excess = len(msg) - max_chars
        if excess <= 0 or not evidence:
            return Composed(evidence, notes, msg)
        longest = max(range(len(evidence)), key=lambda i: len(evidence[i].text))
        text = evidence[longest].text
        base = text[: -len(ELLIPSIS)] if text.endswith(ELLIPSIS) else text
        # keep enough of the passage to drop `excess` chars; the result must be shorter than before
        keep = min(len(base) - excess, len(text) - len(ELLIPSIS) - 1)
        if keep >= MIN_PASSAGE_CHARS:
            evidence[longest].text = base[:keep].rstrip() + ELLIPSIS
        else:
            evidence.pop()  # all passages are at the minimum: drop the least relevant (last) one
