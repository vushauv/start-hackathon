# Database

The `postgres` service of `docker-compose.yml` (PostgreSQL 18 with pgvector), schema in [`schema.sql`](schema.sql):
four tables, what answering a question needs.

| Table | Holds |
|---|---|
| `history_sessions` | which sessions a history has, and each one's date in that history |
| `turns` | session content, once per session, with a keyword-search index |
| `chunks` | overlapping windows of whole sentences over a session, made by `chunk.py` |
| `questions` | the questions, with their dates |

Each split lives in its own Postgres schema (`dev`, `dev_large`, `test`, `final`): dev and dev-large reuse the same
history and question ids for different histories, so they must not share tables.

```sh
cp .env.example .env                                 # once; set the password
docker compose up -d --wait postgres
python3 -m venv .venv && .venv/bin/pip install -r db/requirements.txt
.venv/bin/python db/load.py --split dev              # reads data/dev, creates schema dev
.venv/bin/python db/load.py --split test             # reads data/test, creates schema test
.venv/bin/python db/load.py --split final --data-dir /path/to/final
```

- Connection: the `POSTGRES_*` values of `.env` (`db/settings.py`), or `DATABASE_URL` when set. From code, pick
  the split with `SET search_path TO dev, public` (`public` holds pgvector's types).
- `load.py` is idempotent: what is already loaded is skipped. Chunks are not loaded; `chunk.py` makes them.
- Schema changes: edit `schema.sql`, then `load.py --split <split> --reset` (drops that split's data, embeddings
  included).
- Gold answers stay in the files: score with the challenge's `score.py`.

## Chunks

```sh
.venv/bin/python db/chunk.py --split dev                     # chunk sessions not chunked yet
.venv/bin/python db/chunk.py --split dev --reset             # re-chunk everything (drops embeddings too)
.venv/bin/python db/chunk.py --split dev --show 5            # 5 sample chunks with a question; writes nothing
.venv/bin/python db/chunk.py --split dev --expand 's_...:3' --before 1 --after 1
```

- About 500 tokens a chunk (bge-m3's tokenizer), 200 of overlap, cut only between sentences; a question stays with
  the rest of its turn and the answer when they fit. `--max-tokens`, `--overlap`, `--min-fill`, `--unit-cap` tune it;
  the algorithm is described at the top of `chunk.py`.
- Text is tagged per speaker: `[USER] ...` / `[ASSISTANT] ...`.
- Each chunk has its span in the session (`start_turn:start_char` to `end_turn:end_char`), `idx` and
  `prev_chunk_id`/`next_chunk_id`. To merge neighbours, use `expand()` (or `--expand`): it renders the span from the
  first chunk's start to the last one's end, so overlapping text appears once.
- Embeddings are not made here: `embedded_text` and `embedded_schemed_text` stay NULL for the embedding script, and so
  does `schemed_text`.
