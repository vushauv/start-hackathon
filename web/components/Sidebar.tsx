"use client";

import { useMemo, useState } from "react";
import type { HistoryDetail, Meta } from "@/lib/types";
import { IconClose, IconSearch } from "./icons";
import { typeLabel } from "./format";

interface Props {
  meta: Meta | null;
  historyId: string | null;
  history: HistoryDetail | null;
  loadingHistory: boolean;
  open: boolean;
  onClose: () => void;
  onSelectHistory: (id: string) => void;
  onAskDataset: (q: { id: string; question: string }) => void;
}

export function Sidebar({ meta, historyId, history, loadingHistory, open, onClose, onSelectHistory, onAskDataset }: Props) {
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"histories" | "questions">("histories");

  const histories = useMemo(() => {
    const q = query.trim().toLowerCase();
    const all = meta?.histories ?? [];
    return q ? all.filter((h) => h.id.toLowerCase().includes(q) || h.first_question.toLowerCase().includes(q)) : all;
  }, [meta, query]);

  return (
    <aside className={`sidebar ${open ? "is-open" : ""}`}>
      <div className="sidebar-head">
        <div>
          <div className="wordmark">Recall</div>
          <div className="env">
            <span className={`dot ${meta ? "ok" : ""}`} />
            {meta ? `${meta.split} · ${meta.device}` : "Connecting…"}
          </div>
        </div>
        <button className="btn btn-text btn-icon only-mobile" onClick={onClose} aria-label="Close sidebar">
          <IconClose />
        </button>
      </div>

      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "histories"} onClick={() => setTab("histories")}>
          Histories{meta ? <span className="count">{meta.histories.length}</span> : null}
        </button>
        <button role="tab" aria-selected={tab === "questions"} onClick={() => setTab("questions")}>
          Questions{history ? <span className="count">{history.questions.length}</span> : null}
        </button>
      </div>

      {tab === "histories" ? (
        <>
          <label className="search">
            <IconSearch width={14} height={14} />
            <input className="input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search" />
          </label>
          <ul className="list">
            {histories.map((h) => (
              <li key={h.id}>
                <button
                  className={`list-item ${h.id === historyId ? "is-active" : ""}`}
                  onClick={() => {
                    onSelectHistory(h.id);
                    onClose();
                  }}
                >
                  <span className="list-title">
                    <span>{h.id}</span>
                    <span>{h.questions}</span>
                  </span>
                  <span className="list-sub">{h.first_question}</span>
                </button>
              </li>
            ))}
            {meta && histories.length === 0 ? <li className="list-empty">No matches</li> : null}
          </ul>
        </>
      ) : (
        <ul className="list" style={{ paddingTop: 8 }}>
          {loadingHistory ? <li className="list-empty">Indexing…</li> : null}
          {history?.questions.map((q) => (
            <li key={q.id}>
              <button
                className="list-item"
                onClick={() => {
                  onAskDataset(q);
                  onClose();
                }}
              >
                <span className="list-q">{q.question}</span>
                <span className="list-sub">{q.type ? typeLabel(q.type) : q.id}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
