"use client";

import { useMemo, useState } from "react";
import type { Meta } from "@/lib/types";
import { IconClose, IconSearch } from "./icons";

interface Props {
  meta: Meta | null;
  historyId: string | null;
  recentChats: { historyId: string; question: string; updatedAt: number }[];
  open: boolean;
  onClose: () => void;
  onSelectHistory: (id: string) => void;
}

export function Sidebar({ meta, historyId, recentChats, open, onClose, onSelectHistory }: Props) {
  const [query, setQuery] = useState("");

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

      <label className="search">
        <IconSearch width={14} height={14} />
        <input className="input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search histories" />
      </label>
      {!query.trim() && recentChats.length ? (
        <section className="recent-chats" aria-label="Recent conversations">
          <p className="sidebar-label">Recent chats</p>
          <ul className="recent-list">
            {recentChats.slice(0, 6).map((chat) => (
              <li key={chat.historyId}>
                <button
                  className={`recent-item ${chat.historyId === historyId ? "is-active" : ""}`}
                  onClick={() => {
                    onSelectHistory(chat.historyId);
                    onClose();
                  }}
                >
                  <span>{chat.question}</span>
                  <small className="mono">{chat.historyId}</small>
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      <p className="sidebar-label">All histories{meta ? <span>{meta.histories.length}</span> : null}</p>
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
    </aside>
  );
}
