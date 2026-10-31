# web: Recall, the chat UI for the pipeline

A Next.js (TypeScript) chat over `pipeline chat`. You ask a question about a memory (a dataset history),
and the answer shows its citations, the ground truth (on dev) and the full retrieval trace: embedding,
dense/BM25 → RRF, reranker, type/k/hints, packed evidence and the composed message.

```
browser ──► Next.js (web/, :3000) ──/api/*──► pipeline serve (:8765) ──► retriever + Reactor / Groq / OpenAI
                                      └─/memory-api/*──► FastAPI (:8000) ──► saved Reactor conversations
```

## Run

```sh
# 1. the API (loads bge-small + reranker once; GPU if available)
cd pipeline && uv sync --extra retrieval
uv run --extra retrieval pipeline serve --split dev            # add --device cpu, --port, --provider groq ...

# 2. the UI
cd web && npm install
npm run dev                                                     # http://localhost:3000
```

The UI proxies `/api/*` to `PIPELINE_API_URL` (default `http://127.0.0.1:8765`, see `.env.example`).

## Settings

The sliders button (top right, or the summary pill in the composer) opens a drawer with the `chat` flags,
applied per question and remembered in the browser:

| Drawer | CLI flag |
|---|---|
| Provider / Model | `--provider`, `--model` |
| Max chars / Attempts / Timeout | `--max-chars`, `--attempts`, `--timeout` |
| Dry run | `--dry-run` |
| Reranker / Rerank text / Fusion | `--reranker`, `--rerank-text`, `--fusion` |
| Evidence selection | per-type top-k (chat default) or `--rerank-threshold` |
| Question date | `:date` (free-form questions; dataset questions use their own date) |
| Show composed message | `--show-message` |

## Notes

- The server handles one question at a time (models and Reactor are shared); a second request waits
  and the UI says it is queued.
- Indexing a history is slow the first time on CPU (~80 s); vectors are cached in `retriever/cache/<split>/`.

## Docker

Start the persistent API and UI with `docker compose --profile api up --build`. It exposes the UI at
`http://localhost:3000`, and proxies saved conversation endpoints at `/memory-api/v1/conversations`.
For the full animated retrieval graph, start the GPU trace service too:

```sh
docker compose --profile api --profile trace up --build
```

The trace service needs an NVIDIA-enabled Docker runtime. It persists downloaded models and embeddings in
the `retriever_cache` volume; subsequent starts reuse them.

### Public URL with Cloudflare Tunnel

Before enabling this, revoke any Cloudflare API token accidentally exposed in chat or shell history. Create a
**named, remotely-managed** tunnel in Cloudflare's dashboard, add a public hostname, and set its service to
`http://memory-web:3000`. Copy the tunnel connector token (not an account API token) into the ignored root
`.env` file:

```sh
CLOUDFLARE_TUNNEL_TOKEN=replace-with-the-named-tunnel-token
```

Then run the UI plus tunnel, and add `--profile trace` when the live retrieval graph is wanted:

```sh
docker compose --profile api --profile tunnel up -d
# or: docker compose --profile api --profile trace --profile tunnel up -d
```

The public hostname exposes the application and can spend provider quota. Add a Cloudflare Access policy or
other authentication before sharing it broadly.

## MCP server

The optional `memory-mcp` service exposes the persistent backend through Streamable HTTP MCP at
`http://localhost:8001/mcp`. It has tools to list/create/resume conversations and ask configured dataset questions.

```sh
docker compose --profile api --profile mcp up -d --build
```

The web app also publishes these connection instructions at `/mcp`. To make MCP remotely available, add another
Cloudflare Tunnel public hostname whose service is `http://memory-mcp:8001`, then connect an MCP client to
`https://mcp.your-domain.example/mcp`. Put Cloudflare Access in front of that hostname; do not expose the MCP
answer tool anonymously.
