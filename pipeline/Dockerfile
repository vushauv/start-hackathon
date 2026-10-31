# Build context: repository root (``docker compose --profile api up --build memory-api``).
FROM python:3.12-slim

# The persistent conversation API needs only ``api``. The optional trace
# service enables this at build time so it can load the retriever and reranker.
ARG INSTALL_RETRIEVAL=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir "uv>=0.5"

# The pipeline has an editable local dependency on reactor_service, so both source trees
# must be present before uv resolves and installs the service extra.
COPY pipeline /app/pipeline
COPY reactor_service /app/reactor_service
# ``pipeline serve`` loads the repository-level retriever module at runtime.
# It is only used by the optional trace image, but copying it here keeps both
# Docker targets built from the same Dockerfile.
COPY retriever /app/retriever

WORKDIR /app/pipeline
RUN if [ "$INSTALL_RETRIEVAL" = "1" ]; then \
      uv sync --no-dev --extra api --extra retrieval; \
    else \
      uv sync --no-dev --extra api; \
    fi

EXPOSE 8000

ENTRYPOINT ["/app/pipeline/.venv/bin/pipeline", "api-serve"]
CMD ["--split", "dev", "--data", "/data", "--evidence", "/evidence/evidence.jsonl", "--state", "/state/conversations.jsonl", "--host", "0.0.0.0", "--port", "8000"]
