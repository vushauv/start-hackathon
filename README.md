# Long-term Memory Benchmark Explorer

Tooling around a LongMemEval-style conversational-memory dataset: given a user's
long chat history (~48 sessions), answer questions about them.

**The end-to-end solution is `pipeline/`** ([pipeline/README.md](pipeline/README.md)):
retrieval -> Reactor -> `predictions.jsonl` -> score. It joins the parts below.

| Part | What it is |
|---|---|
| [`retriever/`](retriever/README.md) | hybrid session retriever (bge-small + BM25 + RRF) with a bge-reranker cross-encoder; GPU, run once per split |
| [`embedder/`](embedder/README.md) | the standalone bge-small embedder and the experiment behind the chunking choice |
| [`reactor_service/`](reactor_service/README.md) | Python client for the team's Reactor agent (message template, citations, retries) |
| [`db/`](db/README.md) | PostgreSQL 18 + pgvector schema, loader and sentence-aware chunker (an alternative retrieval backend, not yet wired into the pipeline) |
| [`frontend/`](frontend/README.md) | React dataset explorer |

Services run via Docker Compose: the explorer and PostgreSQL.

## Producing a submission

```sh
cd pipeline && uv sync
uv run pipeline retrieve --split test        # GPU box: uv sync --extra retrieval first
uv run pipeline answer   --split test        # needs reactor_service/.env with the Reactor handout
uv run pipeline predict  --split test        # -> pipeline/runs/test/predictions.jsonl
uv run pipeline score    --split dev         # dev only
```

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

## Configuration

Compose reads settings from `.env`. Create it from the template and set values:

```
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `POSTGRES_USER` | Database superuser name | `postgres` |
| `POSTGRES_PASSWORD` | Database superuser password | `change-me` |
| `POSTGRES_DB` | Database created on first start | `memory` |
| `POSTGRES_PORT` | Host port (bound to `127.0.0.1`) | `5432` |

Credentials are applied only when the data volume is first initialised. After
changing them, reset the volume with `docker compose down -v`.

## Quick start

```
docker compose up --build   # frontend at http://localhost:5173, hot reload
```

Postgres data persists in the `postgres_data` volume. Connect with:

```
psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:$POSTGRES_PORT/$POSTGRES_DB"
```

## Database: load and chunk

With the splits in `data/` and `postgres` running:

```
python3 -m venv .venv && .venv/bin/pip install -r db/requirements.txt
.venv/bin/python db/load.py --split dev      # data/dev -> schema dev
.venv/bin/python db/chunk.py --split dev     # sentence-aware chunks, no embeddings
```

Details in [`db/README.md`](db/README.md).
