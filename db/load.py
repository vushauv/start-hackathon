"""Load one split (histories.jsonl, questions.jsonl) into Postgres, chunks excluded.

    python db/load.py --split dev                         # reads data/dev
    python db/load.py --split dev-large
    python db/load.py --split test --data-dir /path/to/test
    python db/load.py --split final --reset

Each split gets its own Postgres schema (namespace), named after it with '-' as '_': dev, dev_large, test, final.
This is needed because dev and dev-large reuse the same history and question ids for different histories. The
schema's tables come from db/schema.sql, created the first time a split is loaded. Query a split with
``SET search_path TO dev_large, public``.

Loading is idempotent: histories and questions already loaded are skipped, so a split can be loaded again after an
interruption, and a session shared by several histories has its turns stored once. ``--reset`` drops the split's
schema first, chunks and embeddings included.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg import sql

from settings import DSN, ROOT

DATE_FMT = "%Y/%m/%d (%a) %H:%M"
SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"


def read_jsonl(path):
    """Yield dicts from a JSON lines file."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_date(text):
    """A dataset date, '2023/05/20 (Sat) 02:21', as a datetime."""
    return datetime.strptime(text, DATE_FMT)


def copy(cur, table, columns, rows):
    """COPY rows into table; nothing is sent for an empty list."""
    if rows:
        with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as cp:
            for row in rows:
                cp.write_row(row)


def existing(conn, query):
    """The set of values the query returns."""
    with conn.cursor() as cur:
        cur.execute(query)
        return {r[0] for r in cur}


def prepare(conn, schema, reset):
    """Create the split's schema and tables if missing (after dropping them, with reset); point search_path at it."""
    with conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        if reset:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        found = conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (schema,)).fetchone()
        if not found:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        if not found:
            conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    return not found


def load_histories(conn, path, batch):
    """Each history's session dates, and the turns of sessions not stored yet; one transaction per batch."""
    seen_h = existing(conn, "SELECT DISTINCT history_id FROM history_sessions")
    seen_s = existing(conn, "SELECT DISTINCT session_id FROM turns")
    n_hist = 0
    links, turns = [], []

    def flush():
        with conn.transaction(), conn.cursor() as cur:
            copy(cur, "history_sessions", ("history_id", "session_id", "date_raw", "ts"), links)
            copy(cur, "turns", ("turn_id", "session_id", "idx", "role", "content"), turns)
        links.clear()
        turns.clear()

    for h in read_jsonl(path):
        if h["history_id"] in seen_h:
            continue
        for s in h["sessions"]:
            sid = s["session_id"]
            links.append((h["history_id"], sid, s["date"], parse_date(s["date"])))
            if sid not in seen_s:
                seen_s.add(sid)
                turns += [(t["turn_id"], sid, i, t["role"], t["content"].replace("\x00", ""))  # no NUL in Postgres text
                          for i, t in enumerate(s["turns"])]
        n_hist += 1
        if n_hist % batch == 0:
            flush()
            print(f"  {n_hist} histories", file=sys.stderr, end="\r")
    flush()
    print(f"  {n_hist} histories", file=sys.stderr)


def load_questions(conn, path):
    """Questions not loaded yet."""
    seen = existing(conn, "SELECT question_id FROM questions")
    rows = [(q["question_id"], q["history_id"], q["question"], q["question_date"], parse_date(q["question_date"]))
            for q in read_jsonl(path) if q["question_id"] not in seen]
    with conn.transaction(), conn.cursor() as cur:
        copy(cur, "questions", ("question_id", "history_id", "question", "date_raw", "ts"), rows)
    print(f"  {len(rows)} questions", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Load one dataset split into its own Postgres schema (chunks excluded).")
    ap.add_argument("--split", required=True, help="dev, dev-large, test, final, ...: names the Postgres schema")
    ap.add_argument("--data-dir", type=Path, help="folder with histories.jsonl and questions.jsonl "
                                                  "(default: data/<split> in the repository)")
    ap.add_argument("--dsn", default=DSN, help="default: $DATABASE_URL, else POSTGRES_* from .env")
    ap.add_argument("--batch", type=int, default=10, help="histories per transaction")
    ap.add_argument("--reset", action="store_true", help="drop the split's schema first, chunks included")
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", args.split):
        ap.error("--split: lowercase letters, digits, '-' and '_', starting with a letter")
    data = args.data_dir or ROOT / "data" / args.split
    for name in ("histories.jsonl", "questions.jsonl"):
        if not (data / name).exists():
            ap.error(f"{data / name} not found (download the split, or point --data-dir at it)")
    schema = args.split.replace("-", "_")
    with psycopg.connect(args.dsn) as conn:
        created = prepare(conn, schema, args.reset)
        print(f"{data} -> schema {schema}{' (created)' if created else ''}", file=sys.stderr)
        load_histories(conn, data / "histories.jsonl", args.batch)
        load_questions(conn, data / "questions.jsonl")
        conn.execute("ANALYZE history_sessions, turns, questions")


if __name__ == "__main__":
    main()
