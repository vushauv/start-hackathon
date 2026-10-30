"""Offline retrieval metrics against gold evidence session ids.

Answer text is not scored here. That is ``vendor/rxai/score.py``, which needs
Reactor's answers. These metrics say whether the retriever put the gold
sessions where a packed message can still carry them.

* ``recall@k`` / ``precision@k`` use the ranked unique session ids. Precision's
  denominator is ``k``, so a retriever that returns fewer than ``k`` sessions
  is not given credit for the empty slots.
* ``mrr`` is the reciprocal rank of the first gold session.
* ``evidence_f1`` is the official session-set F1 of the passages the packer
  would send (the budget cutoff), on answerable questions. Abstention gold
  has no evidence session, and that F1 is left unset, matching ``score.py``.
* Abstention is a separate decision: no snippet, or a top score at or below
  ``abstain_below``. ``abstention_recall`` is how often that fires on gold
  abstentions. ``false_abstention`` is how often it fires on answerable
  questions.
"""

from __future__ import annotations

from collections import defaultdict

DEFAULT_K = (1, 3, 5, 10)


def evidence_f1(citations: list[str], evidence: list[str]) -> float:
    """F1 of cited vs gold session ids, the same definition as ``score.py``."""

    cited, gold = set(citations), set(evidence)
    hit = len(cited & gold)
    if not hit:
        return 0.0
    precision = hit / len(cited)
    recall = hit / len(gold)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float | None:
    if not gold:
        return None
    hit = len(set(ranked[:k]) & set(gold))
    return hit / len(set(gold))


def precision_at_k(ranked: list[str], gold: list[str], k: int) -> float | None:
    if k < 1 or not gold:
        return None
    hit = len(set(ranked[:k]) & set(gold))
    return hit / k


def mean_reciprocal_rank(ranked: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    gold_ids = set(gold)
    for rank, sid in enumerate(ranked, 1):
        if sid in gold_ids:
            return 1.0 / rank
    return 0.0


def sent_recall(sent: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    return len(set(sent) & set(gold)) / len(set(gold))


def sent_precision(sent: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    if not sent:
        return 0.0
    return len(set(sent) & set(gold)) / len(set(sent))


def decides_abstain(snippets: list[dict], abstain_below: float) -> bool:
    """Abstain when retrieval found nothing strong enough to send."""

    if not snippets:
        return True
    score = snippets[0].get("score")
    if not isinstance(score, (int, float)):
        return True
    return float(score) <= abstain_below


def score_question(
    *,
    question_id: str,
    question_type: str,
    gold_abstain: bool,
    gold_evidence: list[str],
    pred_abstain: bool,
    ranked: list[str],
    sent: list[str],
    top_score: float | None,
    message_chars: int | None,
    k_values: tuple[int, ...] = DEFAULT_K,
) -> dict:
    """Per-question retrieval record. Retrieval metrics are unset when gold abstains."""

    answerable = not gold_abstain
    record = {
        "question_id": question_id,
        "question_type": question_type,
        "gold_abstain": gold_abstain,
        "pred_abstain": pred_abstain,
        "gold_evidence": list(gold_evidence),
        "ranked": list(ranked),
        "sent": list(sent),
        "top_score": top_score,
        "message_chars": message_chars,
        "recall": {},
        "precision": {},
        "mrr": mean_reciprocal_rank(ranked, gold_evidence) if answerable else None,
        "evidence_f1": evidence_f1(sent, gold_evidence) if answerable else None,
        "sent_recall": sent_recall(sent, gold_evidence) if answerable else None,
        "sent_precision": sent_precision(sent, gold_evidence) if answerable else None,
    }
    for k in k_values:
        record["recall"][str(k)] = recall_at_k(ranked, gold_evidence, k) if answerable else None
        record["precision"][str(k)] = precision_at_k(ranked, gold_evidence, k) if answerable else None
    return record


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _column_value(record: dict, column: str):
    if column.startswith("recall@"):
        return record["recall"].get(column.split("@", 1)[1])
    if column.startswith("precision@"):
        return record["precision"].get(column.split("@", 1)[1])
    return record.get(column)


def metric_columns(k_values: tuple[int, ...] = DEFAULT_K) -> list[str]:
    columns = ["n", "n_answerable"]
    columns += [f"recall@{k}" for k in k_values]
    columns += [f"precision@{k}" for k in k_values]
    columns += [
        "mrr",
        "evidence_f1",
        "sent_recall",
        "sent_precision",
        "abstention_n",
        "abstention_recall",
        "false_abstention",
    ]
    return columns


def summarize(records: list[dict], k_values: tuple[int, ...] = DEFAULT_K) -> dict[str, dict]:
    """Per question type, then ``all``. Means skip unset metrics."""

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["question_type"]].append(record)
    ordered = {name: groups[name] for name in sorted(groups)}
    ordered["all"] = list(records)
    columns = metric_columns(k_values)
    rows = {}
    for name, group in ordered.items():
        abstain = [r for r in group if r["gold_abstain"]]
        answerable = [r for r in group if not r["gold_abstain"]]
        row: dict = {
            "n": len(group),
            "n_answerable": len(answerable),
            "abstention_n": len(abstain),
            "abstention_recall": (
                sum(r["pred_abstain"] for r in abstain) / len(abstain) if abstain else None
            ),
            "false_abstention": (
                sum(r["pred_abstain"] for r in answerable) / len(answerable) if answerable else None
            ),
        }
        for column in columns:
            if column in row:
                continue
            row[column] = _mean([value for r in answerable if (value := _column_value(r, column)) is not None])
        rows[name] = row
    return rows


def format_table(rows: dict[str, dict], k_values: tuple[int, ...] = DEFAULT_K) -> str:
    """A fixed-width table. Floats print with three decimals; unset cells are ``-``."""

    columns = metric_columns(k_values)
    names = list(rows)
    type_width = max(len("type"), *(len(name) for name in names)) if names else len("type")

    def cell(value, width: int) -> str:
        if value is None:
            text = "-"
        elif isinstance(value, float):
            text = f"{value:.3f}"
        else:
            text = str(value)
        return f"{text:>{width}}"

    header_widths = []
    for column in columns:
        width = len(column)
        for row in rows.values():
            width = max(width, len(cell(row.get(column), 0).strip()))
        header_widths.append(max(width, len(column)))

    header = f"{'type':<{type_width}}  " + "  ".join(
        f"{column:>{width}}" for column, width in zip(columns, header_widths)
    )
    lines = [header]
    for name, row in rows.items():
        lines.append(
            f"{name:<{type_width}}  "
            + "  ".join(cell(row.get(column), width) for column, width in zip(columns, header_widths))
        )
    return "\n".join(lines)
