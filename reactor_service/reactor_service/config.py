"""Settings loaded from environment / .env / the organizers' JSON handout."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError

_PKG_DIR = Path(__file__).resolve().parent.parent
_ENV_FILES = (".env", str(_PKG_DIR / ".env"))


class ReactorSettings(BaseSettings):
    """Connection settings. Every field maps to ``REACTOR_<NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="REACTOR_",
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(description="https://tNN.api.reactor-hackathon.rxai.dev, no path")
    agent_id: str = Field(description="agt_...")
    access_key: SecretStr = Field(description="rxk_... never logged or committed")
    team_name: str | None = None
    slot: str | None = None
    timeout: float = Field(default=300, gt=0, le=86400)
    max_chars: int = Field(default=16000, gt=0)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        parts = urlsplit(v)
        local = parts.hostname in ("127.0.0.1", "localhost")
        if parts.scheme != "https" and not (parts.scheme == "http" and local):
            raise ValueError("base_url must start with https://")
        if parts.path or parts.query or parts.fragment:
            raise ValueError("base_url must be just https://HOST, with no path")
        if not parts.hostname:
            raise ValueError("base_url has no host")
        return v

    @field_validator("agent_id")
    @classmethod
    def _check_agent(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("agent_id must not be empty")
        return v

    @field_validator("access_key")
    @classmethod
    def _check_key(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("access_key must not be empty")
        return SecretStr(v.get_secret_value().strip())

    # -- derived -------------------------------------------------------------

    @property
    def ws_base(self) -> str:
        return "wss://" + self.base_url[len("https://"):] if self.base_url.startswith("https://") \
            else "ws://" + self.base_url[len("http://"):]

    def history_url(self) -> str:
        return f"{self.base_url}/agents/{self.agent_id}/history"

    def chat_url(self, history_id: str | None = None) -> str:
        from urllib.parse import quote

        url = f"{self.ws_base}/agents/{self.agent_id}/ws/chat"
        if history_id:
            url += "?history=" + quote(history_id, safe="")
        return url

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_key.get_secret_value()}"}

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> "ReactorSettings":
        """Build settings from the organizers' handout JSON.

        Expected keys: ``base_url``, ``agent_id``, ``access_key`` and optionally
        ``team_name``, ``slot``. Environment variables are ignored.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.update(overrides)
        return cls(_env_file=None, **data)  # type: ignore[call-arg]

    @classmethod
    def load(cls, **overrides) -> "ReactorSettings":
        try:
            return cls(**overrides)
        except Exception as e:  # pydantic ValidationError
            raise ConfigError(
                f"Reactor settings are incomplete or invalid: {e}\n"
                "Set REACTOR_BASE_URL, REACTOR_AGENT_ID and REACTOR_ACCESS_KEY "
                "in the environment or in reactor_service/.env (see .env.example)."
            ) from None


@lru_cache(maxsize=1)
def get_settings() -> ReactorSettings:
    """Settings loaded once from env / .env. Clear with ``get_settings.cache_clear()``."""
    return ReactorSettings.load()
