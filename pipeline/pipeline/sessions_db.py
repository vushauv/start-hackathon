"""Read a history's sessions and transcripts from the PostgreSQL schema built by ``db/load.py``.

Optional: without psycopg or a reachable database every method raises ``DBUnavailable`` and the server falls back
to the dataset files. One schema per split (``test``, ``dev``, ...), see ``db/README.md``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = re.compile(r"^[a-z_][a-z0-9_]*$")


class DBUnavailable(RuntimeError):
    pass


def _env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                out[key.strip()] = value.strip().strip("'\"")
    return out


def dsn() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env = {**_env_file(ROOT / ".env"), **os.environ}
    return (f"postgresql://{env.get('POSTGRES_USER', 'postgres')}:{env.get('POSTGRES_PASSWORD', '')}"
            f"@127.0.0.1:{env.get('POSTGRES_PORT', '5432')}/{env.get('POSTGRES_DB', 'memory')}")


class SessionsDB:
    def __init__(self, url: str | None = None):
        self.url = url or dsn()

    def _query(self, split: str, sql: str, args: tuple):
        if not SCHEMA.match(split):
            raise DBUnavailable(f"bad schema name {split!r}")
        try:
            import psycopg
        except ImportError as e:
            raise DBUnavailable("psycopg is not installed (uv sync --extra ui)") from e
        try:
            with psycopg.connect(self.url, connect_timeout=3) as conn:
                conn.execute(f"SET search_path TO {split}, public")
                return conn.execute(sql, args).fetchall()
        except psycopg.Error as e:
            raise DBUnavailable(str(e).splitlines()[0]) from e

    def sessions(self, split: str, history_id: str) -> list[dict]:
        rows = self._query(split, """
            SELECT hs.session_id, hs.date_raw, count(t.turn_id),
                   coalesce((SELECT left(content, 140) FROM turns u
                             WHERE u.session_id = hs.session_id AND u.role = 'user' ORDER BY u.idx LIMIT 1), '')
            FROM history_sessions hs LEFT JOIN turns t ON t.session_id = hs.session_id
            WHERE hs.history_id = %s GROUP BY hs.session_id, hs.date_raw, hs.ts ORDER BY hs.ts, hs.session_id
        """, (history_id,))
        if not rows:
            raise DBUnavailable(f"history {history_id} is not loaded in schema {split}")
        return [{"session_id": r[0], "date": r[1], "n_turns": r[2], "preview": r[3]} for r in rows]

    def session(self, split: str, history_id: str, session_id: str) -> dict | None:
        order = self.sessions(split, history_id)
        ids = [s["session_id"] for s in order]
        if session_id not in ids:
            return None
        turns = self._query(split, "SELECT turn_id, role, content FROM turns WHERE session_id = %s ORDER BY idx",
                            (session_id,))
        i = ids.index(session_id)
        return {"history_id": history_id, "session_id": session_id, "date": order[i]["date"],
                "turns": [{"turn_id": t[0], "role": t[1], "content": t[2]} for t in turns],
                "prev": ids[i - 1] if i > 0 else None, "next": ids[i + 1] if i + 1 < len(ids) else None,
                "index": i, "total": len(ids)}
