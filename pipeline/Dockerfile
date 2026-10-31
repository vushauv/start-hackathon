# Build context: repository root (``docker compose --profile api up --build memory-api``).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir "uv>=0.5"

# The pipeline has an editable local dependency on reactor_service, so both source trees
# must be present before uv resolves and installs the service extra.
COPY pipeline /app/pipeline
COPY reactor_service /app/reactor_service

WORKDIR /app/pipeline
RUN uv sync --no-dev --extra api

EXPOSE 8000

ENTRYPOINT ["/app/pipeline/.venv/bin/pipeline", "api-serve"]
CMD ["--split", "dev", "--data", "/data", "--evidence", "/evidence/evidence.jsonl", "--state", "/state/conversations.jsonl", "--host", "0.0.0.0", "--port", "8000"]
