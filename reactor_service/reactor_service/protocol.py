"""Pure functions: message template, final-answer parsing, citation mapping."""

from __future__ import annotations

import re
import secrets

from .models import Citation, Evidence

ABSTAIN = "INSUFFICIENT_EVIDENCE"
ACK_SUFFIX = "Reply only OK."

_FINAL_RE = re.compile(
    r"^[ \t]*#+[ \t]*final[ \t_-]*answer[ \t]*\[[ \t]*ref[ \t]*:[ \t]*([0-9a-f]{8})[ \t]*\][ \t]*:?",
    re.IGNORECASE | re.MULTILINE,
)
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_LABEL_RE = re.compile(r"\bE(\d+)\b", re.IGNORECASE)


def new_ref() -> str:
    """Fresh 8-hex-digit reference tying a question to its final answer."""
    return secrets.token_hex(4)


def build_message(
    question: str,
    evidence: list[Evidence] | tuple[Evidence, ...] = (),
    *,
    ref: str,
    question_date: str | None = None,
) -> str:
    """Compose the text sent to Reactor for one question."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    lines = [f"Question [ref:{ref}]: {question}"]
    if question_date:
        if " " in question_date or "T" in question_date:
            lines.append(
                f"Question date: {question_date} (wall-clock time of the question, no timezone)."
            )
        else:
            lines.append(f"Question date: {question_date} (treat this as today's date).")
    if evidence:
        lines.append("")
        lines.append("Evidence:")
        for i, p in enumerate(evidence, start=1):
            meta = f"id={p.id}"
            if p.date:
                meta += f", date={p.date}"
            lines.append(f"[E{i}] ({meta})")
            lines.append(p.text.strip())
            lines.append("")
        lines.append(
            "Answer the question from the evidence above and cite the labels you used, "
            "like [E1] or [E2, E3]."
        )
    else:
        lines.append("")
        lines.append("Answer the question from what you remember.")
    lines.append("")
    lines.append(
        f"Finish with one line that starts exactly with `# FINAL ANSWER [ref:{ref}]:` "
        "followed by the answer. "
        f"If the evidence does not answer the question, write "
        f"`# FINAL ANSWER [ref:{ref}]: {ABSTAIN}` instead. "
        "Anything before that line is treated as interim notes."
    )
    return "\n".join(lines)


def final_answers(reply: str) -> dict[str, str]:
    """Map ``ref -> answer segment`` for every ``# FINAL ANSWER [ref:..]:`` in ``reply``.

    A segment runs from its marker to the next marker or end of text. A
    repeated ref keeps its last segment.
    """
    matches = list(_FINAL_RE.finditer(reply))
    out: dict[str, str] = {}
    for n, m in enumerate(matches):
        end = matches[n + 1].start() if n + 1 < len(matches) else len(reply)
        out[m.group(1).lower()] = reply[m.end():end].strip()
    return out


def is_abstention(answer: str) -> bool:
    return ABSTAIN in answer.upper().replace(" ", "_").replace("-", "_")


def citations(
    answer: str, evidence: list[Evidence] | tuple[Evidence, ...]
) -> tuple[list[Citation], list[str]]:
    """Extract ``[E#]`` labels from ``answer`` and map them to evidence ids.

    Returns ``(cited, unknown)`` where ``unknown`` lists labels with no
    matching passage.
    """
    cited: list[Citation] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for group in _BRACKET_RE.findall(answer):
        for num in _LABEL_RE.findall(group):
            label = f"E{int(num)}"
            if label in seen:
                continue
            seen.add(label)
            idx = int(num) - 1
            if 0 <= idx < len(evidence):
                cited.append(Citation(label=label, id=evidence[idx].id))
            else:
                unknown.append(label)
    return cited, unknown


def history_part(text: str, index: int, total: int) -> str:
    """Wrap one chunk of a dataset history for loading into Reactor memory."""
    return (
        f"Part {index} of {total} of my past chats, to remember. {ACK_SUFFIX}\n\n"
        f"{text.strip()}\n\n{ACK_SUFFIX}"
    )
