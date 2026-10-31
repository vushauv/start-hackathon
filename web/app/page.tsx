"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ask, fetchMeta } from "@/lib/api";
import type { Meta, Settings, StreamEvent, Turn } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import { Composer } from "@/components/Composer";
import { Message } from "@/components/Message";
import { SettingsDrawer } from "@/components/SettingsDrawer";
import { IconMenu, IconSliders } from "@/components/icons";
import { shortDate } from "@/components/format";

const SETTINGS_KEY = "recall.settings.v1";
const THREADS_KEY = "recall.threads.v1";

const FALLBACK: Settings = {
  reranker: "base",
  rerank_text: "best2",
  fusion: "fused",
  rerank_threshold: null,
  provider: "reactor",
  model: "",
  max_chars: null,
  attempts: null,
  timeout: null,
  dry_run: false,
  show_message: false,
};

function loadStored(): Partial<Settings> {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

/** Keep prior chat answers, not bulky retrieval traces or hidden reasoning. */
function loadThreads(): Record<string, Turn[]> {
  if (typeof window === "undefined") return {};
  try {
    const value: unknown = JSON.parse(localStorage.getItem(THREADS_KEY) ?? "{}");
    if (!value || typeof value !== "object") return {};
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).flatMap(([historyId, turns]) => {
        if (!Array.isArray(turns)) return [];
        const safe = turns.filter((turn): turn is Turn =>
          !!turn && typeof turn === "object" && typeof (turn as Turn).id === "string" && typeof (turn as Turn).question === "string",
        );
        return safe.length ? [[historyId, safe]] : [];
      }),
    );
  } catch {
    return {};
  }
}

function compactThreads(threads: Record<string, Turn[]>): Record<string, Turn[]> {
  return Object.fromEntries(
    Object.entries(threads).map(([historyId, turns]) => [
      historyId,
      // Avoid filling localStorage with long evidence excerpts. The trace is live-only;
      // restored conversations retain the user question, answer, status, and citations.
      turns.slice(-40).map(({ steps: _steps, waiting: _waiting, gold: _gold, answer, ...turn }) => ({
        ...turn,
        steps: {},
        answer: answer ? { ...answer, reasoning: undefined, reply: undefined } : undefined,
      })),
    ]),
  );
}

function apply(t: Turn, e: StreamEvent): Turn {
  switch (e.type) {
    case "queued":
      return { ...t, phase: "queued" };
    case "start":
      return { ...t, phase: "retrieving", questionDate: e.question_date, historyId: e.history_id };
    case "step":
      return { ...t, phase: "retrieving", steps: { ...t.steps, [e.id]: { ms: e.ms, data: e.data } } };
    case "waiting":
      return { ...t, phase: "waiting", waiting: { who: e.who, expect: e.expect } };
    case "answer":
      return { ...t, answer: e };
    case "gold":
      return { ...t, gold: e };
    case "done":
      return { ...t, phase: "done", totalMs: e.total_ms };
    case "error":
      return { ...t, phase: "error", error: e.message };
  }
}

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [historyId, setHistoryId] = useState<string | null>(null);
  const [threads, setThreads] = useState<Record<string, Turn[]>>(loadThreads);
  const [settings, setSettings] = useState<Settings>(FALLBACK);
  const [questionDate, setQuestionDate] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const turns = (historyId && threads[historyId]) || [];
  const recentChats = useMemo(
    () => Object.entries(threads)
      .filter(([, items]) => items.length > 0)
      .map(([id, items]) => ({ historyId: id, question: items.at(-1)?.question ?? "New conversation", updatedAt: items.at(-1)?.startedAt ?? 0 }))
      .sort((a, b) => b.updatedAt - a.updatedAt),
    [threads],
  );

  const connect = useCallback(() => {
    setMetaError(null);
    fetchMeta()
      .then((m) => {
        setMeta(m);
        setSettings({ ...FALLBACK, ...m.defaults, ...loadStored() });
        setHistoryId((h) => h ?? m.histories[0]?.id ?? null);
      })
      .catch((e: Error) => setMetaError(e.message));
  }, []);

  useEffect(connect, [connect]);

  useEffect(() => {
    if (!meta) return;
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      /* storage unavailable: settings last for this visit only */
    }
  }, [settings, meta]);

  useEffect(() => {
    try {
      localStorage.setItem(THREADS_KEY, JSON.stringify(compactThreads(threads)));
    } catch {
      /* storage quota unavailable: the active chat remains usable */
    }
  }, [threads]);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns.length, turns[turns.length - 1]?.phase, turns[turns.length - 1]?.answer]);

  const run = useCallback(
    async (question: string, questionId: string | null = null) => {
      if (!historyId || busy) return;
      const hid = historyId;
      const id = crypto.randomUUID();
      const turn: Turn = { id, question, questionId, historyId: hid, phase: "retrieving", steps: {}, startedAt: Date.now() };
      const update = (fn: (t: Turn) => Turn) =>
        setThreads((all) => ({ ...all, [hid]: (all[hid] ?? []).map((t) => (t.id === id ? fn(t) : t)) }));
      setThreads((all) => ({ ...all, [hid]: [...(all[hid] ?? []), turn] }));
      setBusy(true);
      const ctrl = new AbortController();
      abort.current = ctrl;
      try {
        await ask(
          { history_id: hid, question, question_id: questionId, question_date: questionId ? undefined : questionDate, settings },
          (e) => update((t) => apply(t, e)),
          ctrl.signal,
        );
        update((t) => (t.phase === "done" || t.phase === "error" ? t : { ...t, phase: "error", error: "stream ended early" }));
      } catch (e) {
        const msg = ctrl.signal.aborted ? "Stopped." : (e as Error).message;
        update((t) => ({ ...t, phase: "error", error: msg }));
      } finally {
        setBusy(false);
        abort.current = null;
      }
    },
    [historyId, busy, questionDate, settings],
  );

  return (
    <div className="app">
      <Sidebar
        meta={meta}
        historyId={historyId}
        recentChats={recentChats}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelectHistory={setHistoryId}
      />

      <main className="main">
        <header className="topbar">
          <button className="btn btn-text btn-icon only-mobile" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
            <IconMenu />
          </button>
          <div className="topbar-title">
            <span className="mono">{historyId ?? "–"}</span>
            {questionDate ? (
              <span className="topbar-meta">
                <button onClick={() => setSettingsOpen(true)} title="Change question date">
                  {shortDate(questionDate.replaceAll("-", "/"))}
                </button>
              </span>
            ) : null}
          </div>
          <div className="topbar-actions">
            <a className="btn btn-text hide-mobile" href="/mcp">
              MCP guide
            </a>
            {turns.length ? (
              <button
                className="btn btn-text hide-mobile"
                disabled={busy}
                onClick={() => historyId && setThreads((all) => ({ ...all, [historyId]: [] }))}
              >
                New chat
              </button>
            ) : null}
            <button className="btn" onClick={() => setSettingsOpen(true)}>
              <IconSliders width={14} height={14} />
              <span className="hide-mobile">Settings</span>
            </button>
          </div>
        </header>

        <div className="scroller" ref={scroller}>
          <div className="column">
            {metaError ? (
              <div className="banner">
                <div>
                  <strong>Pipeline API unreachable</strong>
                  <p className="mono">{metaError}</p>
                </div>
                <button className="btn" onClick={connect}>
                  Retry
                </button>
              </div>
            ) : null}

            {turns.length === 0 ? (
              <div className="intro">
                <h1>What would you like to remember?</h1>
                <p>
                  {historyId
                    ? "Ask a question to search this history. Retrieval and evidence will appear only after you send it."
                    : "Choose a history from the sidebar, then ask a question."}
                </p>
              </div>
            ) : (
              <div className="thread">
                {turns.map((t, i) => (
                  <Message key={t.id} turn={t} isLast={i === turns.length - 1} />
                ))}
              </div>
            )}
          </div>
        </div>

        <Composer
          busy={busy}
          disabled={!historyId}
          settings={settings}
          onSubmit={(t) => run(t)}
          onStop={() => abort.current?.abort()}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </main>

      <SettingsDrawer
        open={settingsOpen}
        meta={meta}
        settings={settings}
        questionDate={questionDate}
        latestDate=""
        onChange={setSettings}
        onQuestionDate={setQuestionDate}
        onReset={() => {
          setSettings({ ...FALLBACK, ...(meta?.defaults ?? {}) });
          setQuestionDate("");
        }}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
