# Long-term Memory Benchmark Explorer

Tooling around a LongMemEval-style conversational-memory dataset: given a user's
long chat history (~48 sessions), answer questions about them.

## Dataset

Three JSON-L files live in the repo root. They are **git-ignored** and must be
obtained separately and placed next to this README.

| File | Rows | Content |
|---|---|---|
| `histories.jsonl` | 470 | `history_id` -> `sessions[]` -> `turns[]` (~249 MB) |
| `questions.jsonl` | 526 | `question_id`, `history_id`, `question`, `question_date` |
| `gold.jsonl` | 526 | `question_type`, `answer`, `abstain`, `aliases`, `evidence` (session IDs) |

Question types: multi-session, temporal-reasoning, single-session-assistant,
knowledge-update, single-session-user. 30 questions are abstention cases.

## Data setup

The dataset is published as release assets (`data-v1`) on
[RxAI-dev/memory-challenge](https://github.com/RxAI-dev/memory-challenge/releases).
Download, extract and clean up inside a `data/` directory:

```
mkdir -p data && cd data

BASE=https://github.com/RxAI-dev/memory-challenge/releases/download/data-v1
wget $BASE/dev.tar.gz
wget $BASE/dev-large.tar.gz   # optional, ~744 MB
wget $BASE/test.tar.gz

for f in *.tar.gz; do tar -xzf "$f"; done   # extract
rm -f *.tar.gz                              # remove the compressed archives

cd ..
```

The release assets are gzip-compressed (`.tar.gz`). If an archive is ever
published as `.tar.xz`, extract it with `tar -xJf` instead.

## Branches

- `main` - stable
- `dev` - integration
- `feature/frontend` - React dataset explorer (see `frontend/`)

## Quick start

```
docker compose up --build   # frontend at http://localhost:5173, hot reload
```

## Retrieval eval (`memtest`)

`memtest` is the retrieval test harness. It loads a split, runs a retriever, and
scores the session ids it returns against dev gold. Reactor is only involved
in the optional online command, and final answers there still come from the
vendored client (`vendor/rxai/client.py`).

```
pip install -r requirements.txt
python -m memtest offline --retriever bm25 --split dev --limit 50
pytest
```

`offline` prints a per-type table and writes `runs/<UTC timestamp>/report.json`
plus `details.jsonl`. The report records the retriever, split, limit, session
cap, and character budget, so two runs can be compared by those fields.
`--limit` keeps the first N questions in file order. In data-v1 those rows are
grouped by question type, so a small limit covers one type.

| Piece | What it does |
| --- | --- |
| Loaders | Stream JSONL. Histories are indexed by byte offset and read one line at a time, so dev-large (about 744MB) is not held in memory. |
| Retriever | `retrieve(question, history)` returns `[{session_id, date, text, score}]`, best first. `bm25` is Okapi BM25 over user and assistant turns (`rank_bm25`), one snippet per session. Register another class with `memtest.retriever.register_retriever`. |
| Metrics | On answerable questions: recall@k, precision@k (denominator is k), MRR, and evidence F1 of the sessions that fit in the message we would send. Abstention recall and false abstention are reported separately, including gold abstentions. |
| Packer | Fills the Reactor client's 16,000-character message, using the vendored `build_message`. Passages keep the session id, the session date, and role labels. Assistant turns are dropped only after lower-scoring turns. |
| Online | `python -m memtest online --retriever bm25 --split dev --limit 5`. Needs `REACTOR_ACCESS_KEY`, `REACTOR_BASE_URL`, and `REACTOR_AGENT_ID` (flags `--base-url` and `--agent-id` override the last two). The key is never a command-line argument. If any of the three is unset, the command prints a skip message and exits 0 without calling the client. Results append to `results.jsonl` and a rerun skips question ids already there. `predictions.jsonl` is written by the vendored `predictions.py`. |
| Official score | `python -m memtest score --gold data/dev/gold.jsonl --predictions predictions.jsonl` runs `vendor/rxai/score.py`. |

The kit files under `vendor/rxai/` are copied from [RxAI-dev/memory-challenge](https://github.com/RxAI-dev/memory-challenge) (see `vendor/rxai/UPSTREAM.md`). Tests use `tests/fixtures/dev/` and do not download the release.
