"""Where the database is: $DATABASE_URL, else the POSTGRES_* settings of the environment or the repository's .env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_env(path):
    """KEY=value lines of a .env file; comments and blank lines skipped."""
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                values[key.strip()] = value.strip().strip("'\"")
    return values


def database_url():
    """The connection string docker compose's postgres service answers on."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env = {**read_env(ROOT / ".env"), **os.environ}
    user, password = env.get("POSTGRES_USER", "postgres"), env.get("POSTGRES_PASSWORD", "")
    return (f"postgresql://{user}:{password}@127.0.0.1:{env.get('POSTGRES_PORT', '5432')}/"
            f"{env.get('POSTGRES_DB', 'memory')}")


DSN = database_url()
