import type { HistoryDetail, Meta, Settings, StreamEvent } from "./types";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `${res.status} ${res.statusText}`);
  return body as T;
}

export const fetchMeta = () => getJSON<Meta>("/api/meta");
export const fetchHistory = (id: string) => getJSON<HistoryDetail>(`/api/history/${encodeURIComponent(id)}`);

export interface AskRequest {
  history_id: string;
  question: string;
  question_id?: string | null;
  question_date?: string;
  settings: Settings;
}

/** POST /api/ask and call `onEvent` for every NDJSON line as it arrives. */
export async function ask(req: AskRequest, onEvent: (e: StreamEvent) => void, signal?: AbortSignal) {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `${res.status} ${res.statusText}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onEvent(JSON.parse(line) as StreamEvent);
    }
  }
  if (buf.trim()) onEvent(JSON.parse(buf) as StreamEvent);
}
