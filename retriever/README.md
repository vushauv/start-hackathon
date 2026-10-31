# Retriever + reranker (RxAI memory challenge)

Hybrid session retriever (bge-small dense + BM25, RRF) with a bge-reranker-base cross-encoder. It builds the `evidence_<split>.jsonl` that gets sent to Reactor. On all 526 dev questions: R@5 0.972, R@k 0.964. See RESULTS.md. The embedder lives in `embedder/` (PR #7).

Install: `pip install -r retriever/requirements.txt` (CUDA torch recommended; a 4 GB GPU is enough).
Data: set `RETRIEVER_DATA_ROOT` to the folder containing `dev/` and `test/` (or pass `--data <split dir>`). Embedding cache: `cache/{dev,test}/<history_id>.v1.npz` (vectors only; texts are re-chunked from the history on load).

```bash
cd retriever
export RETRIEVER_DATA_ROOT=/path/to/data
PY=python

# evidence for every question of a split (default config = bge-reranker-base, top-2 chunks, fused with RRF)
$PY build_evidence.py --split test            # -> evidence_test.jsonl  (~2 min cold, ~30 s cached)
$PY build_evidence.py --split dev             # -> evidence_dev.jsonl   (~13 min cold index, ~3 min cached)
$PY build_evidence.py --split test --reranker v2-m3     # alternative reranker
$PY build_evidence.py --split test --reranker none      # RRF only

# full evaluation on all 526 dev questions (RRF only vs base vs v2-m3, best1/best2, rerank-only vs fused)
$PY eval_retriever.py                         # -> eval_results.json (~26 min cold, ~13 min with cached index)
$PY make_results.py                           # -> RESULTS.md
```

## Pipeline (retriever.py)
1. **Index per history.** B is user-turn chunks. F is user+assistant pairs in 256-token windows with stride 192. BM25 runs on the F chunks. Embeddings use bge-small-en-v1.5 in fp16, with the bge query prefix on questions.
2. **First stage.** RRF (k=60) over the session rankings from B, F and BM25, where a session's score is the max over its chunks. This gives the top 20 candidates. Inside each session, chunks are ordered by the RRF of dense-F and BM25.
3. **Rerank.** A cross-encoder scores (question, top-2 chunks of the session, max 512 tokens). The final order is the RRF of the rerank rank and the first-stage rank.
4. **Per-type k.** Guessed by `guess_type()` (rule-based): SSU/SSA 3, KU 4, temporal 4, multi-session 6.
5. **Evidence.** Sessions are packed with a water-filling budget of 14,000 characters or less, best session first. Inside a session, the top-ranked chunks come first, then the remaining turns in order, trimmed.
6. **Hint.** For guessed temporal and KU questions: session dates, days before question_date, chronological order, day gaps, and the most recent session.

Output line: `{question_id, history_id, question, question_date, guessed_type, evidence:[{id, date, text}], hint, candidates}`.
`candidates` holds all 20 session ids in final order. You can use them to cite or to widen k.

API: `from retriever import Retriever; r = Retriever(split_dir, cache_dir); rows = r.run()`.
