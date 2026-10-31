"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ask, fetchHistory, fetchMeta } from "@/lib/api";
import type { HistoryDetail, Meta, Settings, StreamEvent, Turn } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import { Composer } from "@/components/Composer";
import { Message } from "@/components/Message";
import { SettingsDrawer } from "@/components/SettingsDrawer";
import { IconMenu, IconSliders } from "@/components/icons";
import { shortDate, typeLabel } from "@/components/format";

const SETTINGS_KEY = "recall.settings.v1";

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
  const [history, setHistory] = useState<HistoryDetail | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [threads, setThreads] = useState<Record<string, Turn[]>>({});
  const [settings, setSettings] = useState<Settings>(FALLBACK);
  const [questionDate, setQuestionDate] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const turns = (historyId && threads[historyId]) || [];

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
    if (!historyId) return;
    let live = true;
    setLoadingHistory(true);
    setHistory(null);
    setQuestionDate("");
    fetchHistory(historyId)
      .then((h) => live && setHistory(h))
      .catch((e: Error) => live && setMetaError(e.message))
      .finally(() => live && setLoadingHistory(false));
    return () => {
      live = false;
    };
  }, [historyId]);

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
        history={history}
        loadingHistory={loadingHistory}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelectHistory={setHistoryId}
        onAskDataset={(q) => run(q.question, q.id)}
      />

      <main className="main">
        <header className="topbar">
          <button className="btn btn-text btn-icon only-mobile" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
            <IconMenu />
          </button>
          <div className="topbar-title">
            <span className="mono">{historyId ?? "–"}</span>
            {history ? (
              <span className="topbar-meta">
                {history.sessions} sessions ·{" "}
                <button onClick={() => setSettingsOpen(true)} title="Change question date">
                  {questionDate ? shortDate(questionDate.replaceAll("-", "/")) : shortDate(history.question_date)}
                </button>
              </span>
            ) : loadingHistory ? (
              <span className="topbar-meta">Indexing…</span>
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
                Clear
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
                <h1>Ask about this history</h1>
                <p>
                  {history
                    ? `${history.sessions} sessions of past conversations. Answers cite the sessions they use.`
                    : loadingHistory
                      ? "Indexing sessions…"
                      : "Select a history to begin."}
                </p>
                {history?.questions.length ? (
                  <>
                    <p className="section-label">From the dataset</p>
                    <ul className="qlist">
                      {history.questions.slice(0, 6).map((q) => (
                        <li key={q.id}>
                          <button disabled={busy} onClick={() => run(q.question, q.id)}>
                            <span className="q-text">{q.question}</span>
                            <small>{q.type ? typeLabel(q.type) : q.id}</small>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </>
                ) : null}
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
          disabled={!history}
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
        latestDate={history?.question_date ?? ""}
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
