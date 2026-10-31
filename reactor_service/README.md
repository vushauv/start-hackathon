# Reactor service

Plain Python functions for talking to the team's RxAI Reactor agent
(the [memory-challenge](https://github.com/RxAI-dev/memory-challenge) kit).
Inputs and outputs are pydantic models. Credentials come from `.env`.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
cd reactor_service
uv sync
cp .env.example .env     # then paste the values from the organizers' handout
uv run pytest            # offline tests
```

To use the package from elsewhere in the repo, add it as a path dependency
(`uv add ../reactor_service`) or run scripts with `uv run --project reactor_service`.

## Configuration

The handout JSON maps to these variables (prefix `REACTOR_`):

| Variable | Handout key | Description | Default |
|---|---|---|---|
| `REACTOR_BASE_URL` | `base_url` | `https://tNN.api.reactor-hackathon.rxai.dev`, no path | required |
| `REACTOR_AGENT_ID` | `agent_id` | `agt_...` | required |
| `REACTOR_ACCESS_KEY` | `access_key` | `rxk_...` secret, never commit or paste it | required |
| `REACTOR_TEAM_NAME` | `team_name` | informational | |
| `REACTOR_SLOT` | `slot` | informational | |
| `REACTOR_TIMEOUT` | | seconds to wait for an answer | `300` |
| `REACTOR_MAX_CHARS` | | client-side message limit (server: 32000) | `16000` |

`.env` is looked up in the current directory, then in `reactor_service/`.
You can also load the handout file directly and skip `.env`:

```
from reactor_service import ReactorSettings
s = ReactorSettings.from_json("creds.json")
answer = reactor.ask("...", evidence, settings=s)
```

Every function accepts `settings=`; without it the cached `.env` settings are used.

## Usage

```
import reactor_service as reactor

# 1. One Reactor history per dataset history. Never reset between questions
#    of the same dataset history.
h = reactor.new_history()

# 2. Evidence = excerpts your retrieval found. id is the session_id.
evidence = [
    {"id": "s_07de", "date": "2023/05/20 (Sat) 02:21",
     "text": "user: We booked the hall for May.\nassistant: Great."},
    {"id": "s_81c2", "date": "2023/05/27 (Sat) 10:05",
     "text": "user: The wedding moved to June, the hall is the same."},
]

# 3. Ask. Blocks for tens of seconds. Pass question_date exactly as in questions.jsonl.
a = reactor.ask(
    "Which month is the wedding?",
    evidence,
    question_date="2023-06-01",
    history_id=h.history_id,
    question_id="q_123",
)

a.answer       # 'June [E2]'
a.abstained    # False; True when the agent wrote INSUFFICIENT_EVIDENCE
a.cited_ids    # ['s_81c2']  evidence ids the answer cites, in order
a.citations    # [Citation(label='E2', id='s_81c2')]
a.unknown_citations  # labels the agent cited but you never sent
a.uncited      # evidence was sent but nothing was cited
a.latency_s, a.retries, a.other_replies, a.late_replies

a.to_jsonl()   # one line in the kit's results.jsonl format
```

### Functions

| Function | What it does | Returns |
|---|---|---|
| `new_history()` | `POST /agents/{id}/history`, fresh memory | `History(history_id, started_at, running, restarted)` |
| `ask(question, evidence=(), *, question_date, history_id, question_id, timeout)` | builds the `[E1]..[En]` message with a fresh ref, waits for `# FINAL ANSWER`, maps citations | `Answer` |
| `send_text(text, *, history_id, timeout)` | sends raw text, returns the first reply, no answer parsing | `TextReply(reply, cycle, latency_s, retries, ...)` |
| `load_history(parts, *, history_id, on_part)` | sends each part as "Part i of n ... Reply only OK." and waits for a reply before the next | `list[TextReply]` |
| `split_history(sessions, *, max_chars)` | renders dataset sessions into parts that fit the limit, oldest first | `list[str]` |
| `build_message(question, evidence, *, question_date)` | the exact text `ask` would send, to check sizes | `str` |

Async twins `anew_history`, `aask`, `asend_text`, `aload_history` exist for
asyncio callers. Calling a sync function inside a running event loop raises
`ReactorError` pointing to them.

Evidence items can be `Evidence` models or plain dicts with `id`, `text` and
optional `date`.

### Loading a whole history into memory

```
import json, reactor_service as reactor

history = next(json.loads(l) for l in open("../histories.jsonl") if '"h_42"' in l)
h = reactor.new_history()
parts = reactor.split_history(history["sessions"])     # ~125 parts at 16k chars
reactor.load_history(parts, history_id=h.history_id,
                     on_part=lambda i, n, r: print(f"{i}/{n} {r.reply[:20]}"))
# citations still come only from the evidence passed to ask()
```

## Rules of the API

- One history per dataset history. Start it with `new_history()`, ask all of
  that history's questions in it, never reuse it for another dataset history.
- One question at a time per agent. Run questions sequentially.
- Messages over `REACTOR_MAX_CHARS` are rejected locally (`MessageTooLarge`).
  Send excerpts, not whole sessions.
- The agent can browse public URLs. Never put the key in one.
- Reactor answers asynchronously, typically within tens of seconds. The
  default wait is 300 s.

## Errors

All exceptions subclass `reactor_service.ReactorError`.

| Exception | Cause | Fix |
|---|---|---|
| `ConfigError` | missing/invalid `REACTOR_*` values | fill in `.env` or use `from_json` |
| `MessageTooLarge` | message over `max_chars` | shorter or fewer excerpts |
| `ConnectFailed` (`.status` 401/403) | wrong key or agent id, or agent stopped | check creds, then ask support |
| `ConnectFailed` (`.status` 404) | base URL has a path | use `https://HOST` only |
| `ConnectFailed` (`.status` 409/503) | new history already starting / agent still stopping | retry shortly |
| `ConnectFailed` (no status) | host does not resolve | check handout; API may not be open yet |
| `Refused` (`.info.code`) | server refused the message, see table below | |
| `TimedOut` | no final answer within `timeout` | raise `timeout`, or shorter evidence |
| `Closed` | server closed the socket without answering | retry; if persistent, `new_history()` |

Refusal codes (`Refused.info.code`). `forwarded` is always `false`, meaning the
question never reached the agent:

| Code | Meaning | Action |
|---|---|---|
| `history_required` | no history yet | `new_history()` |
| `history_changed` | a newer history exists, see `.info.current_history_id` | ask there or start another |
| `history_resetting` | a new history is starting | retry in a moment |
| `busy` | another chat with this agent is open | retried automatically (1 s to 8 s backoff) |
| `previous_pending` | previous question not yet answered | retried automatically; if persistent, `new_history()` |
| `too_large` | over the server limit (`.info.limit`, `.info.size`) | fewer or shorter passages |

`Refused`, `TimedOut` and `Closed` carry `.other_replies` (interim notes) and
`.late_replies` (final answers to earlier questions that arrived late).
