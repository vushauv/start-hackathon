# Frontend

React + Vite explorer for the JSON-L dataset.

## Docker (hot reload)

From the repo root, with the `.jsonl` files in place:

```
docker compose up --build    # http://localhost:5173
docker compose down
```

Source is bind-mounted into the container, so edits hot-reload (file watching uses
polling because bind mounts don't reliably emit fs events). The three `.jsonl` files are mounted
read-only at `/data`, and the port is published on `127.0.0.1` only (it's a dev server). After changing `package.json`, rebuild with `--build`.

## Local

```
npm install
npm run dev          # http://localhost:5173
DATA_DIR=/path/to/data npm run dev   # if the .jsonl files aren't in the repo root
```

## Lazy loading

The `.jsonl` files are never downloaded whole. A small Vite plugin (`vite.config.js`)
serves them under `/data/` with HTTP Range support, plus a `.index` endpoint that
returns each line's byte offset/length and id (one streaming scan, ~0.5 s for the 249 MB
histories file, cached). The browser (`src/lib/jsonl.js`) then:

- loads `questions`/`gold` in 100-line chunks, only as far as the current page needs
  (filters and search load the rest on demand);
- fetches a single history with one Range request when a question is selected;
- renders session turns only when a session is expanded.
