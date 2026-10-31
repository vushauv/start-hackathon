"""Pure functions: message template, final-answer parsing, citation mapping."""

from __future__ import annotations

import json
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
_ROLE_LINE_RE = re.compile(r"^(user|assistant):\s*", re.IGNORECASE | re.MULTILINE)


def new_ref() -> str:
    """Fresh 8-hex-digit reference tying a question to its final answer."""
    return secrets.token_hex(4)


def timestamp_turns(text: str, session_date: str | None) -> str:
    """Render benchmark speaker lines as ``[role] [session timestamp]`` for the LLM."""
    timestamp = session_date or "timestamp unavailable"
    return _ROLE_LINE_RE.sub(lambda match: f"[{match.group(1).lower()}] [{timestamp}] ", text.strip())


def build_message(
    question: str,
    evidence: list[Evidence] | tuple[Evidence, ...] = (),
    *,
    ref: str,
    question_date: str | None = None,
    notes: str | None = None,
) -> str:
    """Compose the text sent to Reactor for one question.

    ``notes`` is an optional free-text block (hints, answer-style rules) placed
    after the evidence and before the final-answer instruction.
    """
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    today = question_date or "unknown (no benchmark question date was supplied)"
    lines = [
        f"Today is: {today}. Use this benchmark question date as the sole reference for temporal reasoning.",
        f"Question [ref:{ref}]: {question}",
    ]
    if question_date:
        if " " in question_date or "T" in question_date:
            lines.append(
                f"Question date: {question_date} (the moment the benchmark question is asked: treat it as now; "
                "use this date for temporal reasoning)."
            )
        else:
            lines.append(
                f"Question date: {question_date} (treat this as today's date; do not use the system timestamp "
                "for temporal reasoning)."
            )
    if evidence:
        lines.append("")
        lines.append("Evidence:")
        for i, p in enumerate(evidence, start=1):
            meta = f"id={p.id}"
            if p.date:
                meta += f", date={p.date}"
            lines.append(f"[E{i}] ({meta})")
            lines.append(timestamp_turns(p.text, p.date))
            lines.append("")
        lines.append(
            "Answer the question from the evidence above and cite the labels you used, "
            "like [E1] or [E2, E3]."
        )
    else:
        lines.append("")
        lines.append("Answer the question from what you remember.")
    if notes and notes.strip():
        lines.append("")
        lines.append("Notes:")
        lines.append(notes.strip())
    lines.append("")
    lines.append(
        f"Finish with one line that starts exactly with `# FINAL ANSWER [ref:{ref}]:` "
        "followed by valid JSON with exactly `answer` and `citations` fields. "
        "`citations` must be an array of only the evidence session IDs that directly support the final answer; "
        "do not cite retrieved but unused sessions. "
        f"For example: `# FINAL ANSWER [ref:{ref}]: {{\"answer\":\"June\",\"citations\":[\"s_abc\"]}}`. "
        f"If the evidence does not answer the question, write "
        f"`# FINAL ANSWER [ref:{ref}]: {{\"answer\":\"{ABSTAIN}\",\"citations\":[]}}` instead. "
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


def final_output(
    answer: str, evidence: list[Evidence] | tuple[Evidence, ...]
) -> tuple[str, list[Citation], list[str]]:
    """Parse an answer plus its model-selected evidence session IDs.

    The preferred final-answer payload is ``{"answer":"June", "citations":["s_2"]}``.
    Every supplied ID is validated against the evidence the model actually saw.
    Legacy free-text answers with ``[E#]`` labels remain supported.
    """
    try:
        payload = json.loads(answer)
    except (TypeError, json.JSONDecodeError):
        cited, unknown = citations(answer, evidence)
        return answer, cited, unknown

    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str) or not isinstance(payload.get("citations"), list):
        cited, unknown = citations(answer, evidence)
        return answer, cited, unknown

    cited: list[Citation] = []
    unknown: list[str] = []
    seen: set[str] = set()
    by_id = {item.id: (index, item) for index, item in enumerate(evidence, start=1)}
    for raw in payload["citations"]:
        if not isinstance(raw, str):
            unknown.append(str(raw))
            continue
        label_match = re.fullmatch(r"\s*E(\d+)\s*", raw, re.IGNORECASE)
        if label_match:
            index = int(label_match.group(1))
            item = evidence[index - 1] if 0 < index <= len(evidence) else None
            label = f"E{index}"
        else:
            hit = by_id.get(raw)
            index, item = hit if hit else (0, None)
            label = f"E{index}" if item else raw
        if item is None:
            unknown.append(raw)
        elif item.id not in seen:
            seen.add(item.id)
            cited.append(Citation(label=label, id=item.id))
    return payload["answer"], cited, unknown


def history_part(text: str, index: int, total: int) -> str:
    """Wrap one chunk of a dataset history for loading into Reactor memory."""
    return (
        f"Part {index} of {total} of my past chats, to remember. {ACK_SUFFIX}\n\n"
        f"{text.strip()}\n\n{ACK_SUFFIX}"
    )
