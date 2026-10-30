# Long-term Memory Benchmark Explorer

Tooling around a LongMemEval-style conversational-memory dataset: given a user's
long chat history (~48 sessions), answer questions about them. Services run via
Docker Compose: a React dataset explorer (`frontend/`) and PostgreSQL 18.

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
