"use client";

import { useEffect } from "react";
import type { Meta, Provider, Settings } from "@/lib/types";
import { IconClose } from "./icons";
import { isoDate, shortDate } from "./format";

interface Props {
  open: boolean;
  meta: Meta | null;
  settings: Settings;
  questionDate: string; // "" = latest session of the history
  latestDate: string;
  onChange: (s: Settings) => void;
  onQuestionDate: (d: string) => void;
  onReset: () => void;
  onClose: () => void;
}

function Segmented<T extends string>({ value, options, onChange }: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup">
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Switch({ label, hint, checked, onChange }: { label: string; hint: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="switch-row">
      <div>
        {label}
        <small>{hint}</small>
      </div>
      <button role="switch" aria-checked={checked} aria-label={label} className="toggle" onClick={() => onChange(!checked)} />
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      {children}
    </div>
  );
}

const numOrNull = (v: string) => (v.trim() === "" ? null : Number(v));

export function SettingsDrawer({ open, meta, settings: s, questionDate, latestDate, onChange, onQuestionDate, onReset, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => onChange({ ...s, [k]: v });
  const pd = meta?.provider_defaults[s.provider];

  return (
    <>
      <div className={`scrim ${open ? "is-open" : ""}`} onClick={onClose} />
      <aside className={`drawer ${open ? "is-open" : ""}`} aria-hidden={!open} aria-label="Settings">
        <header className="drawer-head">
          <h2>Settings</h2>
          <button className="btn btn-text btn-icon" onClick={onClose} aria-label="Close settings">
            <IconClose />
          </button>
        </header>

        <div className="drawer-body">
          <section>
            <h3>Answering</h3>
            <Group label="Provider">
              <Segmented<Provider>
                value={s.provider}
                options={[
                  { value: "reactor", label: "Reactor" },
                  { value: "groq", label: "Groq" },
                  { value: "openai", label: "OpenAI" },
                ]}
                onChange={(v) => onChange({ ...s, provider: v, model: "", max_chars: null })}
              />
            </Group>
            {s.provider !== "reactor" ? (
              <Field label="Model">
                <input className="input mono" value={s.model} placeholder={pd?.model} onChange={(e) => set("model", e.target.value)} />
              </Field>
            ) : null}
            <div className="row">
              <Field label="Max chars">
                <input
                  className="input mono"
                  type="number"
                  min={1000}
                  step={1000}
                  value={s.max_chars ?? ""}
                  placeholder={String(pd?.max_chars ?? "")}
                  onChange={(e) => set("max_chars", numOrNull(e.target.value))}
                />
              </Field>
              <Field label="Attempts">
                <input
                  className="input mono"
                  type="number"
                  min={1}
                  max={10}
                  value={s.attempts ?? ""}
                  placeholder={s.provider === "reactor" ? "3" : "6"}
                  onChange={(e) => set("attempts", numOrNull(e.target.value))}
                />
              </Field>
              <Field label="Timeout (s)">
                <input
                  className="input mono"
                  type="number"
                  min={5}
                  value={s.timeout ?? ""}
                  placeholder={s.provider === "reactor" ? "env" : "180"}
                  onChange={(e) => set("timeout", numOrNull(e.target.value))}
                />
              </Field>
            </div>
            <Switch label="Dry run" hint="Retrieval only, no model call" checked={s.dry_run} onChange={(v) => set("dry_run", v)} />
          </section>

          <section>
            <h3>Retrieval</h3>
            <Group label="Reranker">
              <Segmented
                value={s.reranker}
                options={(meta?.rerankers ?? ["base", "v2-m3", "none"]).map((r) => ({ value: r, label: r === "none" ? "Off" : r }))}
                onChange={(v) => set("reranker", v)}
              />
            </Group>
            <div className="row">
              <Group label="Rerank text">
                <Segmented
                  value={s.rerank_text}
                  options={[
                    { value: "best1", label: "best1" },
                    { value: "best2", label: "best2" },
                  ]}
                  onChange={(v) => set("rerank_text", v)}
                />
              </Group>
              <Group label="Fusion">
                <Segmented
                  value={s.fusion}
                  options={[
                    { value: "fused", label: "Fused" },
                    { value: "rerank", label: "Rerank" },
                  ]}
                  onChange={(v) => set("fusion", v)}
                />
              </Group>
            </div>
            <Group label="Evidence selection">
              <Segmented
                value={s.rerank_threshold == null ? "topk" : "threshold"}
                options={[
                  { value: "topk", label: "Top-k by type" },
                  { value: "threshold", label: "Score threshold" },
                ]}
                onChange={(v) => set("rerank_threshold", v === "topk" ? null : -5.5)}
              />
            </Group>
            {s.rerank_threshold != null ? (
              <div className="slider">
                <input
                  type="range"
                  min={-10}
                  max={2}
                  step={0.5}
                  value={s.rerank_threshold}
                  aria-label="Reranker score threshold"
                  onChange={(e) => set("rerank_threshold", Number(e.target.value))}
                />
                <span className="mono">≥ {s.rerank_threshold.toFixed(1)}</span>
              </div>
            ) : null}
          </section>

          <section>
            <h3>Context</h3>
            <Field label="Question date">
              <div className="date-row">
                <input
                  className="input mono"
                  type="date"
                  value={questionDate || isoDate(latestDate)}
                  onChange={(e) => onQuestionDate(e.target.value)}
                />
                <button className="btn" disabled={!questionDate} onClick={() => onQuestionDate("")}>
                  Reset
                </button>
              </div>
            </Field>
            <p className="field-foot" style={{ margin: 0 }}>
              Default: latest session ({shortDate(latestDate) || "–"}). Dataset questions use their own date.
            </p>
          </section>

          <section>
            <h3>Debug</h3>
            <Switch
              label="Show composed message"
              hint="Adds the full prompt to the trace"
              checked={s.show_message}
              onChange={(v) => set("show_message", v)}
            />
          </section>
        </div>

        <footer className="drawer-foot">
          <button className="btn btn-text" onClick={onReset}>
            Reset all
          </button>
          <button className="btn btn-primary" onClick={onClose}>
            Done
          </button>
        </footer>
      </aside>
    </>
  );
}
