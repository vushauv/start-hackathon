"""Memory-challenge answering pipeline.

Stages (each one a module, each one replaceable):

    data        read a split (questions, histories, gold)
    retrieval   question -> evidence passages (GPU retriever, or a precomputed file)
    prompting   evidence + hints -> the notes block and a message that fits the limit
    answering   message -> Reactor final answer (one Reactor history per dataset history)
    postprocess Reactor answer -> {abstain, answer, citations}
    runner      ties the stages together, resumable, writes results.jsonl / predictions.jsonl
    scoring     the organizers' score.py on dev
"""

from .models import Evidence, Prediction, QuestionResult, Retrieved
from .postprocess import Policy, to_prediction
from .runner import run, write_predictions

__all__ = [
    "Evidence", "Prediction", "QuestionResult", "Retrieved",
    "Policy", "to_prediction", "run", "write_predictions",
]
