"""Small, dependency-free checks for the retrieval selection policy."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "retriever"))
import retriever as R  # noqa: E402


def _ordered(n=6):
    return [{"sid": f"s_{i}"} for i in range(n)]


def test_temporal_selection_retains_ranked_minimum_below_threshold():
    ordered = _ordered()
    scores = {c["sid"]: -10.0 - i for i, c in enumerate(ordered)}
    assert [c["sid"] for c in R.select_evidence(ordered, scores, "temporal-reasoning", -5.5)] == [
        "s_0", "s_1", "s_2"
    ]


def test_single_session_selection_remains_precision_first():
    ordered = _ordered()
    scores = {c["sid"]: -10.0 - i for i, c in enumerate(ordered)}
    assert [c["sid"] for c in R.select_evidence(ordered, scores, "single-session-user", -5.5)] == ["s_0"]
