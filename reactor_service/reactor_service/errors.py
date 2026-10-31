"""Exceptions raised by the Reactor client."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RefusalInfo


class ReactorError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(ReactorError):
    """Credentials or base URL are missing or malformed."""


class MessageTooLarge(ReactorError):
    """The message exceeds the client-side ``max_chars`` limit."""

    def __init__(self, size: int, limit: int) -> None:
        self.size = size
        self.limit = limit
        super().__init__(
            f"the message is {size} characters, over the limit of {limit}; "
            "send fewer or shorter excerpts"
        )


class ConnectFailed(ReactorError):
    """The HTTP or WebSocket connection was refused (401/403/404, DNS, ...)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        self.status = status
        super().__init__(message)


class NoAnswer(ReactorError):
    """The question was sent but no answer came back."""

    def __init__(
        self,
        message: str,
        *,
        other_replies: list[str] | None = None,
        late_replies: list[str] | None = None,
    ) -> None:
        self.other_replies = other_replies or []
        self.late_replies = late_replies or []
        super().__init__(message)


class TimedOut(NoAnswer):
    """The overall timeout expired before a final answer arrived."""


class Closed(NoAnswer):
    """The server closed the connection without answering."""


class Refused(NoAnswer):
    """The server refused the message with an ``error`` frame.

    ``info.code`` is one of: ``history_required``, ``history_changed``,
    ``history_resetting``, ``busy``, ``previous_pending``, ``too_large``.
    """

    def __init__(self, info: RefusalInfo, **kw) -> None:
        self.info = info
        text = info.text or info.code or "refused"
        super().__init__(f"refused ({info.code}): {text}", **kw)

    @property
    def retryable(self) -> bool:
        return self.info.code in ("busy", "previous_pending") and self.info.forwarded is False
