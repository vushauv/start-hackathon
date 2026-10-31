"use client";

import { useEffect, useRef, useState } from "react";
import type { Settings } from "@/lib/types";
import { IconArrowUp, IconSliders, IconStop } from "./icons";

interface Props {
  busy: boolean;
  disabled: boolean;
  settings: Settings;
  onSubmit: (text: string) => void;
  onStop: () => void;
  onOpenSettings: () => void;
}

export function Composer({ busy, disabled, settings, onSubmit, onStop, onOpenSettings }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  const send = () => {
    const t = text.trim();
    if (!t || busy || disabled) return;
    onSubmit(t);
    setText("");
  };

  const summary = [
    settings.provider === "reactor" ? "Reactor" : settings.model || settings.provider,
    settings.reranker === "none" ? "no rerank" : settings.reranker,
    settings.rerank_threshold != null ? `≥ ${settings.rerank_threshold}` : "top-k",
  ].join(" · ");

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          ref={ref}
          rows={1}
          value={text}
          disabled={disabled}
          placeholder={disabled ? "Select a history" : "Ask a question"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        <div className="composer-bar">
          <button type="button" className="config" onClick={onOpenSettings} title="Settings">
            <IconSliders width={13} height={13} />
            {summary}
            {settings.dry_run ? <span className="warn">· dry run</span> : null}
          </button>
          {busy ? (
            <button type="button" className="btn btn-icon send" onClick={onStop} aria-label="Stop">
              <IconStop width={14} height={14} />
            </button>
          ) : (
            <button type="submit" className="btn btn-primary btn-icon send" disabled={!text.trim() || disabled} aria-label="Send">
              <IconArrowUp width={14} height={14} />
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
