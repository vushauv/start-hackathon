#!/usr/bin/env bash
# Interactive pipeline REPL. Usage: ./chat.sh [--split dev|test] [--history HID] [--dry-run] [--device cuda|cpu]
set -euo pipefail
cd "$(dirname "$0")"
export PIPELINE_DATA_ROOT="${PIPELINE_DATA_ROOT:-$HOME/reactor-test/labeler/data}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
# reuse the already-built embedding caches if the repo has none
if [ -z "${CHAT_CACHE:-}" ] && [ -d "$HOME/reactor-test/retriever/cache" ]; then CHAT_CACHE="$HOME/reactor-test/retriever/cache"; fi
split=dev; prev=""
for x in "$@"; do [ "$prev" = "--split" ] && split="$x"; prev="$x"; done
extra=()
if [ -n "${CHAT_CACHE:-}" ] && [[ " $* " != *" --cache "* ]]; then extra=(--cache "$CHAT_CACHE/$split"); fi
exec uv run --extra retrieval pipeline chat "${extra[@]}" "$@"
