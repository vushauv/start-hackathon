# Long-term Memory Benchmark Explorer

Tooling around a LongMemEval-style conversational-memory dataset: given a user's
long chat history (~48 sessions), answer questions about them.

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

## Branches

- `main` - stable
- `dev` - integration
- `feature/frontend` - React dataset explorer (see `frontend/`)
