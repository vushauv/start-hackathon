"""Call-style interface to the RxAI Reactor agent.

    import reactor_service as reactor

    h = reactor.new_history()
    a = reactor.ask("When did the wedding move?", evidence, history_id=h.history_id)
    print(a.answer, a.cited_ids)
"""

from .api import (
    aask,
    aload_history,
    anew_history,
    asend_text,
    ask,
    build_message,
    load_history,
    new_history,
    send_text,
    split_history,
)
from .config import ReactorSettings, get_settings
from .errors import (
    Closed,
    ConfigError,
    ConnectFailed,
    MessageTooLarge,
    NoAnswer,
    ReactorError,
    Refused,
    TimedOut,
)
from .models import Answer, AskRequest, Citation, Evidence, History, RefusalInfo, TextReply

__all__ = [
    # functions
    "new_history", "ask", "send_text", "load_history", "build_message", "split_history",
    "anew_history", "aask", "asend_text", "aload_history",
    # config
    "ReactorSettings", "get_settings",
    # models
    "Answer", "AskRequest", "Citation", "Evidence", "History", "RefusalInfo", "TextReply",
    # errors
    "ReactorError", "ConfigError", "ConnectFailed", "MessageTooLarge",
    "NoAnswer", "Refused", "TimedOut", "Closed",
]
