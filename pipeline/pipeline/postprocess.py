"""Post-processing stage: a Reactor result -> the submission line {abstain, answer, citations}.

Everything here is mechanical: labels are stripped the way the kit's predictions.py does, and citations are
the agent's own, filtered to real session ids. The only decisions are the ``Policy`` knobs, all of them about
what to do when the agent gave nothing usable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Prediction, QuestionResult

_LABELS = re.compile(r"\s*\[\s*E\d+(?:\s*[,;]\s*E\d+)*\s*\]")


def strip_labels(answer: str) -> str:
    """The answer without its ``[E#]`` / ``[E1, E3]`` labels, spacing tidied (same as the kit)."""
    text = _LABELS.sub("", answer)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Policy:
    """What to submit when the agent did not answer cleanly.

    * ``fallback_citations``: an answered question whose answer cites nothing gets the first n sent
      passages as citations (evidence F1 is scored on every answerable question; the top passage is
      the best bet).
    * ``abstain_citations``: an abstention keeps the agent's citations if it made any, else the first n
      sent passages. For a truly unanswerable question citations are not scored, so this only helps
      when the abstention was wrong.
    * ``abstain_on_no_answer``: a question with no final answer (timeout, closed, refused after retries)
      is submitted as an abstention; otherwise as an empty answer (scored wrong either way, but an
      abstention is right for the ~8% unanswerable).
    """

    fallback_citations: int = 1
    abstain_citations: int = 1
    abstain_on_no_answer: bool = True


def _dedupe(ids: list[str], known: set[str] | None) -> list[str]:
    out: list[str] = []
    for i in ids:
        if i and i not in out and (known is None or i in known):
            out.append(i)
    return out


def to_prediction(r: QuestionResult, policy: Policy = Policy(), known: set[str] | None = None) -> Prediction:
    """``known`` = the session ids of the question's history; citations outside it are dropped."""
    cited = _dedupe(r.cited_ids, known)
    sent = _dedupe(r.sent_ids, known)
    if r.status == "no-answer":
        if policy.abstain_on_no_answer:
            return Prediction(question_id=r.question_id, abstain=True, answer="",
                              citations=sent[: policy.abstain_citations])
        return Prediction(question_id=r.question_id, abstain=False, answer="", citations=sent[:1])
    if r.status == "abstained":
        return Prediction(question_id=r.question_id, abstain=True, answer="",
                          citations=cited or sent[: policy.abstain_citations])
    answer = strip_labels(r.answer_raw)
    if not cited:
        cited = sent[: policy.fallback_citations]
    return Prediction(question_id=r.question_id, abstain=False, answer=answer, citations=cited)
