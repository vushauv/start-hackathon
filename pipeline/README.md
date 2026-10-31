# pipeline: the end-to-end answering system

Joins the parts of this repo into one run that produces `predictions.jsonl` for a split:

```
questions.jsonl ─► retrieval ─► evidence.jsonl ─► prompting ─► answering (Reactor) ─► results.jsonl ─► postprocess ─► predictions.jsonl ─► score (dev)
                   retriever/                     notes block,                         one line per        abstain / answer /
                   (GPU, once per split)          fit to 16k chars                     question, resumable citations policy
```

Each box is a module in `pipeline/pipeline/` with one job and a small interface, so any of them can be replaced
without touching the others (a Postgres-backed retriever, another prompt, another abstention rule).

| Module | Does | Replace it when |
|---|---|---|
| `data.py` | finds a split's files, reads questions/gold, loads only the needed histories | data layout changes |
| `retrieval.py` | `Retrieval` protocol. `LocalRetrieval` wraps `retriever/retriever.py` (GPU). `FileRetrieval` reads a saved `evidence.jsonl` | a new retriever (e.g. over `db/` chunks) |
| `prompting.py` | the **notes** block (answer-style rules, per-type rules, the retriever's date hints) and `compose()` which shrinks evidence until the message fits `--max-chars` | prompt experiments |
| `answering.py` | `Answerer` protocol. `ReactorAnswerer`: one Reactor history per dataset history, retries, history refresh on `history_changed`. `DryRunAnswerer`: no network | preloading a history into Reactor, several agents in parallel |
| `postprocess.py` | Reactor answer -> `{abstain, answer, citations}` under a `Policy` (citation fallbacks, what to do with no answer) | abstention rules, citation rules |
| `runner.py` | the loop; appends `results.jsonl` per question, skips what is done; `write_predictions()` | |
| `scoring.py` + `vendor/score.py` | the organizers' `score.py`, verbatim | they publish a new one |
| `cli.py` | `pipeline retrieve / answer / predict / score / run` | |
| `models.py` | the records passed between stages (`Retrieved`, `QuestionResult`, `Prediction`) | |

Retrieval is the only stage that needs a GPU and torch. It runs once per split and writes a file; everything
after it runs anywhere (a laptop with just the Reactor credentials). That is why the run is split in two.

## Setup

```sh
cd pipeline
uv sync                      # answer/predict/score: pydantic + reactor_service (path dependency)
uv sync --extra retrieval    # GPU box only: torch, sentence-transformers, transformers, numpy, scipy
cp ../reactor_service/.env.example ../reactor_service/.env   # then fill in the Reactor handout (REACTOR_*)
uv run pytest                # offline tests against a fake Reactor server
```

Data: the split directories from the challenge release (`data/dev`, `data/test`, ...). The default root is
`$PIPELINE_DATA_ROOT`, else `$RETRIEVER_DATA_ROOT`, else `<repo>/data`; `--data DIR` overrides. The repo root
is a fallback for dev (the three JSONL files next to the top README).

## Run

All outputs go to `pipeline/runs/<split>/` (git-ignored): `evidence.jsonl`, `results.jsonl`, `predictions.jsonl`,
`score_details.jsonl`. Every path can be overridden (`--evidence`, `--results`, `--predictions`).

```sh
# 1. GPU box: evidence for every question of the split (~2 min test, ~13 min dev cold; cached embeddings in retriever/cache)
uv run pipeline retrieve --split test                     # --reranker base|v2-m3|none, --device cuda
# Equivalent: python retriever/build_evidence.py --split test --out pipeline/runs/test/evidence.jsonl

# 2. Any box: check sizes without sending, then ask Reactor (sequential, ~10-60 s per question)
uv run pipeline answer --split test --dry-run --show      # prints max/mean message size and the first message
uv run pipeline answer --split test                       # appends runs/test/results.jsonl; rerun = resume

# 3. Turn results into the submission, then score (dev only)
uv run pipeline predict --split test                      # -> runs/test/predictions.jsonl
uv run pipeline score   --split dev
uv run pipeline run     --split dev --limit 30            # all steps that apply, on the first 30 questions
```

Selection flags for trying things: `--limit N`, `--question-ids q_a q_b`, `--history-ids h_x`.

### Resuming and retrying

`results.jsonl` gets one line per question the moment Reactor answers (or gives up). Running `answer` again
asks only the questions missing from it. To ask a question again, delete its line, or just run again: a later
line for the same `question_id` wins in `predict`. A rerun starts a new Reactor history for every dataset
history with questions left, which is fine because every question carries its own evidence.

## What is sent to Reactor

`reactor_service.build_message` renders the question, its date, the passages as `[E1]..[En]` with `id` and
`date`, then a **Notes** block (added for this pipeline; optional argument `notes=` of `ask()`), then the
final-answer instruction. The notes contain:

- General rules: one short definite answer (the judge counts hedging, alternatives and negation as wrong);
  the user's own facts, not those of friends/colleagues (the test split's near misses); abstain with
  `INSUFFICIENT_EVIDENCE` when the fact for the user is not there; cite what was used.
- Per-type rules keyed on the retriever's `guessed_type` (knowledge-update: latest session wins; temporal:
  use the given dates and the precomputed day counts; single-session-assistant: assistant turns; multi-session:
  count only the user's own items).
- The retriever's `hint` lines: session dates, days before the question date, chronological order, day gaps.
  Date arithmetic is done in Python there, not by the model.

Edit `GENERAL_RULES` / `TYPE_RULES` in `prompting.py` to change the prompt; `pipeline answer --dry-run --show`
prints the result.

`compose()` keeps the message under `--max-chars` (16,000 default, the kit's limit; the server allows 32,000)
by trimming the longest passage first (never below 200 characters), then dropping the last, least relevant one.

## How predictions are formed (`postprocess.Policy`)

| Reactor said | `abstain` | `answer` | `citations` |
|---|---|---|---|
| a final answer with `[E#]` | false | answer with labels stripped (as the kit's `predictions.py`) | the cited passages' session ids, in order, no repeats, only ids of this history |
| a final answer citing nothing | false | same | the first `--fallback-citations` (1) passages sent |
| `INSUFFICIENT_EVIDENCE` | true | `""` | what it cited, else the first `--abstain-citations` (1) passages sent |
| nothing (timeout / closed / refused after `--attempts` tries) | true (`--empty-on-no-answer` for `""` instead) | `""` | first passage sent |
| question missing from results | true | `""` | `[]` |

Why citations on abstention: evidence F1 is scored on every answerable question, including those we abstained
on, and on unanswerable ones citations are ignored, so the top passage can only help. Nothing else is touched:
no hand edits, and the answer text is Reactor's.

## Scoring on dev

`pipeline score --split dev` prints the organizers' table (correct / unjudged / abstention recall / false
abstention / evidence F1, per type and overall) and writes per-question details to `runs/dev/score_details.jsonl`.
"Unjudged" answers are those the string matcher cannot decide; the official run sends them to an LLM judge.

Retriever quality on dev (all 526 questions, see `retriever/RESULTS.md`): R@5 0.972, R@k 0.964 with the
default `base|best2|fused` config, so most of the remaining error is in answering, not in finding sessions.

## Extending

- **Another retriever**: implement `retrieve(questions) -> list[Retrieved]` (see `retrieval.py`). Evidence ids
  must be session ids. Save with `save_evidence()` and the rest of the pipeline is unchanged.
- **Abstain on weak retrieval**: `QuestionResult` carries `candidates` and `sent_ids`; add a rule in
  `postprocess.to_prediction` or a wrapper around the `Answerer`.
- **Preloading a whole history into Reactor**: `reactor_service.split_history` + `load_history` in an
  `Answerer.start_history` override; the evidence per question stays, so citations still work.
- **Several agents**: run one `pipeline answer` per agent with disjoint `--history-ids` and separate
  `--results` files, then `predict` over a merged file (or point `--results` at each and concatenate).
- **Resource notes**: `answer`/`predict`/`score` need no GPU and little RAM (histories are loaded only to
  validate citation ids; `--no-check-sessions` skips even that). `retrieve` needs a CUDA GPU with ~2 GB free.

## Tests

`uv run pytest` runs offline: unit tests of prompting and post-processing, and an end-to-end test of the loop
against a fake Reactor (HTTP history endpoint + WebSocket chat) covering answers, abstention, uncited answers,
timeouts, resume and scoring.

## Interactive mode: `pipeline chat`

A REPL that shows every step for one question: query embedding, the top-10 sessions from dense-B /
dense-F / BM25, the RRF top 20, reranker scores and what moved, guessed type + k + date hint, the
evidence passages, and (unless `--dry-run`) Reactor's full reply, final answer and citations.
On dataset questions the gold answer is shown and gold evidence sessions are marked with ★.

```bash
cd pipeline
./chat.sh --split dev --dry-run          # retrieval only, nothing sent to Reactor
./chat.sh --split test                   # real Reactor calls (one at a time; ~40-120 s each)
./chat.sh --split dev --history h_15a8be367c6b --device cpu
# equivalent: PIPELINE_DATA_ROOT=... uv run --extra retrieval pipeline chat --split dev --cache ../retriever/cache/dev
```

`chat.sh` sets `PIPELINE_DATA_ROOT` (default `~/reactor-test/labeler/data`) and reuses
`~/reactor-test/retriever/cache/<split>` if present. Needs the `retrieval` extra (torch).

Commands inside the REPL:

| input | effect |
|---|---|
| `:list` | list the split's histories |
| `:h <history_id>` | switch history (indexes it; cached embeddings are reused) |
| `:q <question_id>` | load a dataset question (switches history, sets its date) |
| `:date YYYY-MM-DD` | set the question date (default: latest session of the history) |
| `:exit` / Ctrl-D | quit |
| anything else | ask it about the current history |

Other flags: `--reranker base|v2-m3|none`, `--fusion fused|rerank`, `--show-message` (print the full
message sent), `--max-chars`. Reactor serves one request per agent: don't use chat without `--dry-run`
while a `pipeline answer` run is going.
