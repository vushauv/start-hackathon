# Test Submission

The final artifact is `pipeline/runs/test/submission-openai-gpt6-astra/predictions.jsonl`.
It is generated from the test split with hybrid retrieval, reranking, and the OpenAI
`gpt-6-astra` model. The input evidence is reused from `pipeline/runs/test/evidence.jsonl`.

## Reproduce

```sh
cd pipeline
uv sync

# OPENAI_API_KEY is read from pipeline/.env.openai or the environment.
uv run pipeline answer \
  --split test --data ../test_dataset \
  --provider openai --model gpt-6-astra --workers 8 --verify-answers \
  --evidence runs/test/evidence.jsonl \
  --results runs/test/submission-openai-gpt6-astra/results.jsonl

uv run pipeline predict \
  --split test --data ../test_dataset --provider openai \
  --evidence runs/test/evidence.jsonl \
  --results runs/test/submission-openai-gpt6-astra/results.jsonl \
  --predictions runs/test/submission-openai-gpt6-astra/predictions.jsonl
```

`results.jsonl` checkpoints each completed answer. Re-running the answer command
resumes incomplete work. The prediction stage emits one object per test question:
`question_id`, `abstain`, `answer`, and `citations`. Abstentions have an empty
citation array; answered rows have one validated session ID, selected by the model
or falling back to the highest-ranked supplied session.
