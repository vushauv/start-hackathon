"use client";

import { useState } from "react";
import { STEP_ORDER, type StepId, type StepMap, type Turn } from "@/lib/types";
import { IconChevron } from "./icons";
import { ms, num, shortDate, typeLabel } from "./format";

const TITLES: Record<StepId, string> = {
  embedding: "Query embedding",
  first_stage: "Dense + BM25 → RRF",
  rerank: "Rerank",
  selection: "Type and hints",
  evidence: "Evidence",
  compose: "Message",
};

function summary<K extends StepId>(id: K, d: StepMap[K]): string {
  switch (id) {
    case "embedding": {
      const e = d as StepMap["embedding"];
      return `${e.dim}-d`;
    }
    case "first_stage": {
      const f = d as StepMap["first_stage"];
      return `${f.candidates.length} candidates`;
    }
    case "rerank": {
      const r = d as StepMap["rerank"];
      return `${r.reranker.replace("BAAI/", "")}, ${r.order.filter((o) => o.selected).length} kept`;
    }
    case "selection": {
      const s = d as StepMap["selection"];
      return `${typeLabel(s.guessed_type)}, ${s.rule}`;
    }
    case "evidence": {
      const e = d as StepMap["evidence"];
      return `${e.passages.length} passages, ${num(e.chars)} chars`;
    }
    case "compose": {
      const c = d as StepMap["compose"];
      return `${num(c.chars)} / ${num(c.limit)} chars`;
    }
  }
  return "";
}

const Gold = ({ on }: { on: boolean }) =>
  on ? (
    <span className="gold-mark" title="Gold evidence">
      ★
    </span>
  ) : null;

function Embedding({ d }: { d: StepMap["embedding"] }) {
  const max = Math.max(...d.head.map(Math.abs), 1e-6);
  return (
    <>
      <dl className="kv">
        <dt>Model</dt>
        <dd className="mono">{d.model}</dd>
        <dt>Norm</dt>
        <dd className="mono">{d.norm.toFixed(4)}</dd>
      </dl>
      <div className="vec" aria-label="First 8 components">
        {d.head.map((x, i) => (
          <div key={i} className="vec-cell" title={x.toFixed(4)}>
            <span className={`vec-bar ${x < 0 ? "neg" : ""}`} style={{ height: `${(Math.abs(x) / max) * 50}%` }} />
          </div>
        ))}
        <span className="vec-more mono">+{d.dim - 8}</span>
      </div>
    </>
  );
}

function FirstStage({ d }: { d: StepMap["first_stage"] }) {
  return (
    <>
      <div className="channels">
        {d.channels.map((c) => (
          <div key={c.name} className="channel">
            <h5>{c.label}</h5>
            <ol>
              {c.top.slice(0, 5).map((t) => (
                <li key={t.sid}>
                  <span className="mono sid">
                    {t.sid}
                    <Gold on={t.gold} />
                  </span>
                  <span className="mono muted">{t.score.toFixed(3)}</span>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </div>
      <table className="grid">
        <thead>
          <tr>
            <th>#</th>
            <th>Session</th>
            <th>Date</th>
            <th className="r">RRF</th>
            <th className="r">B</th>
            <th className="r">F</th>
            <th className="r">BM25</th>
          </tr>
        </thead>
        <tbody>
          {d.candidates.map((c, i) => (
            <tr key={c.sid}>
              <td className="muted">{i + 1}</td>
              <td className="mono">
                {c.sid}
                <Gold on={c.gold} />
              </td>
              <td className="muted">{shortDate(c.date)}</td>
              <td className="r mono">{c.rrf.toFixed(4)}</td>
              <td className="r muted">{c.ranks.dense_B ?? "–"}</td>
              <td className="r muted">{c.ranks.dense_F ?? "–"}</td>
              <td className="r muted">{c.ranks.bm25 ?? "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Rerank({ d }: { d: StepMap["rerank"] }) {
  const scores = d.order.map((o) => o.score ?? 0);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  return (
    <>
      <p className="detail-note">
        Fusion {d.fusion}, text {d.rerank_text}. Highlighted rows are sent.
      </p>
      <table className="grid">
        <thead>
          <tr>
            <th>#</th>
            <th>Session</th>
            <th>Date</th>
            <th>Score</th>
            <th className="r">RRF #</th>
            <th className="r">Δ</th>
          </tr>
        </thead>
        <tbody>
          {d.order.map((o, i) => (
            <tr key={o.sid} className={o.selected ? "is-selected" : "is-dim"}>
              <td>{i + 1}</td>
              <td className="mono">
                {o.sid}
                <Gold on={o.gold} />
              </td>
              <td>{shortDate(o.date)}</td>
              <td>
                {o.score != null ? (
                  <span className="score">
                    <i>
                      <span style={{ width: `${hi > lo ? 4 + ((o.score - lo) / (hi - lo)) * 96 : 100}%` }} />
                    </i>
                    <em className="mono">{o.score.toFixed(2)}</em>
                  </span>
                ) : (
                  "–"
                )}
              </td>
              <td className="r">{o.rrf_rank}</td>
              <td className={`r mono ${o.moved > 0 ? "up" : o.moved < 0 ? "down" : ""}`}>
                {o.moved > 0 ? `↑${o.moved}` : o.moved < 0 ? `↓${-o.moved}` : "–"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Selection({ d }: { d: StepMap["selection"] }) {
  return (
    <>
      <dl className="kv">
        <dt>Guessed type</dt>
        <dd>
          {typeLabel(d.guessed_type)}
          {d.gold_type ? (
            <span className={d.gold_type === d.guessed_type ? "ok" : "warn"}>
              {d.gold_type === d.guessed_type ? "matches gold" : `gold: ${typeLabel(d.gold_type)}`}
            </span>
          ) : null}
        </dd>
        <dt>Selection</dt>
        <dd>
          {d.rule}, {d.k} session{d.k === 1 ? "" : "s"}
        </dd>
        <dt>Question date</dt>
        <dd className="mono">{d.question_date || "–"}</dd>
      </dl>
      {d.hint ? <pre className="pre">{d.hint}</pre> : null}
    </>
  );
}

function EvidenceList({ d }: { d: StepMap["evidence"] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <>
      <div className="meter">
        <span style={{ width: `${Math.min(100, (d.chars / d.cap) * 100)}%` }} />
      </div>
      <p className="detail-note">
        {num(d.chars)} of {num(d.cap)} chars
      </p>
      <div className="passages">
        {d.passages.map((p) => (
          <div key={p.label} className="passage">
            <button onClick={() => setOpen(open === p.label ? null : p.label)} aria-expanded={open === p.label}>
              <span className="cite">{p.label}</span>
              <span className="mono">
                {p.id}
                <Gold on={p.gold} />
              </span>
              <span className="muted">{shortDate(p.date)}</span>
              <span className="muted r">{num(p.chars)}</span>
            </button>
            {open === p.label ? <pre className="pre">{p.text}</pre> : <p className="passage-preview">{p.text.slice(0, 180)}</p>}
          </div>
        ))}
      </div>
    </>
  );
}

function Compose({ d }: { d: StepMap["compose"] }) {
  return (
    <>
      <div className="meter">
        <span style={{ width: `${Math.min(100, (d.chars / d.limit) * 100)}%` }} />
      </div>
      <dl className="kv">
        <dt>Size</dt>
        <dd>
          {num(d.chars)} of {num(d.limit)} chars, {d.passages} passages, notes {num(d.notes_chars)}
        </dd>
        <dt>Retrieval time</dt>
        <dd>{ms(d.retrieval_ms)}</dd>
      </dl>
      {d.message ? <pre className="pre pre-tall">{d.message}</pre> : null}
    </>
  );
}

function Detail({ id, data }: { id: StepId; data: StepMap[StepId] }) {
  switch (id) {
    case "embedding":
      return <Embedding d={data as StepMap["embedding"]} />;
    case "first_stage":
      return <FirstStage d={data as StepMap["first_stage"]} />;
    case "rerank":
      return <Rerank d={data as StepMap["rerank"]} />;
    case "selection":
      return <Selection d={data as StepMap["selection"]} />;
    case "evidence":
      return <EvidenceList d={data as StepMap["evidence"]} />;
    case "compose":
      return <Compose d={data as StepMap["compose"]} />;
  }
}

function nodeState(turn: Turn, id: StepId): "done" | "active" | "pending" {
  if (turn.steps[id]) return "done";
  if (turn.phase === "retrieving" && !STEP_ORDER.find((step) => !turn.steps[step])) return "pending";
  return turn.phase === "retrieving" && STEP_ORDER.find((step) => !turn.steps[step]) === id ? "active" : "pending";
}

function GraphNode({
  id,
  label,
  detail,
  turn,
  selected,
  onSelect,
}: {
  id: StepId;
  label: string;
  detail: string;
  turn: Turn;
  selected: boolean;
  onSelect: (id: StepId) => void;
}) {
  const state = nodeState(turn, id);
  return (
    <button
      className={`graph-node is-${state} ${selected ? "is-selected" : ""}`}
      disabled={!turn.steps[id]}
      onClick={() => onSelect(id)}
      aria-pressed={selected}
      title={turn.steps[id] ? `Open ${label} details` : `${label} has not run yet`}
    >
      <span className="graph-node-dot" />
      <span className="graph-node-label">{label}</span>
      <span className="graph-node-detail mono">{detail}</span>
    </button>
  );
}

function RetrievalMap({ turn, step, onSelect }: { turn: Turn; step: StepId | null; onSelect: (id: StepId) => void }) {
  const first = turn.steps.first_stage?.data;
  const rerank = turn.steps.rerank?.data;
  const evidence = turn.steps.evidence?.data;
  const compose = turn.steps.compose?.data;
  const kept = rerank?.order.filter((item) => item.selected).length;

  return (
    <div className="retrieval-map" aria-label="Live retrieval data-flow graph">
      <div className="graph-origin">
        <span className="graph-origin-kicker">MEMORY QUERY</span>
        <strong>{turn.question}</strong>
      </div>
      <span className="graph-link graph-link-main" aria-hidden="true" />
      <GraphNode id="embedding" label="Embed" detail={turn.steps.embedding ? `${turn.steps.embedding.data.dim}d` : "waiting"} turn={turn} selected={step === "embedding"} onSelect={onSelect} />
      <span className="graph-link graph-link-main" aria-hidden="true" />
      <div className="graph-vault-wrap">
        <span className="graph-vault-title">MEMORY VAULT</span>
        <GraphNode id="first_stage" label="Hybrid search" detail={first ? `${first.candidates.length} candidates` : "dense + BM25"} turn={turn} selected={step === "first_stage"} onSelect={onSelect} />
        <div className="vault-branches" aria-hidden="true">
          <span className={turn.steps.first_stage ? "is-live" : ""}>Dense</span>
          <span className={turn.steps.first_stage ? "is-live" : ""}>BM25</span>
          <span className={turn.steps.first_stage ? "is-live" : ""}>RRF</span>
        </div>
      </div>
      <span className="graph-link graph-link-main" aria-hidden="true" />
      <GraphNode id="rerank" label="Rerank" detail={kept != null ? `${kept} kept` : "waiting"} turn={turn} selected={step === "rerank"} onSelect={onSelect} />
      <span className="graph-link graph-link-main" aria-hidden="true" />
      <GraphNode id="selection" label="Decision" detail={turn.steps.selection ? typeLabel(turn.steps.selection.data.guessed_type) : "waiting"} turn={turn} selected={step === "selection"} onSelect={onSelect} />
      <span className="graph-link graph-link-main" aria-hidden="true" />
      <GraphNode id="evidence" label="Evidence" detail={evidence ? `${evidence.passages.length} passages` : "waiting"} turn={turn} selected={step === "evidence"} onSelect={onSelect} />
      <span className="graph-link graph-link-main" aria-hidden="true" />
      <GraphNode id="compose" label="LLM context" detail={compose ? `${num(compose.chars)} chars` : "waiting"} turn={turn} selected={step === "compose"} onSelect={onSelect} />
    </div>
  );
}

export function Trace({ turn, defaultOpen }: { turn: Turn; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [step, setStep] = useState<StepId | null>(null);
  const done = STEP_ORDER.filter((s) => turn.steps[s]).length;
  const active = turn.phase === "retrieving" ? STEP_ORDER.find((s) => !turn.steps[s]) : undefined;
  const total = turn.steps.compose?.data.retrieval_ms;

  return (
    <section className={`trace ${open ? "is-open" : ""}`}>
      <button className="trace-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <IconChevron className="chev" width={12} height={12} />
        Retrieval graph
        <span className="push mono">{total != null ? ms(total) : `${done}/${STEP_ORDER.length}`}</span>
      </button>
      {open ? (
        <>
          <RetrievalMap turn={turn} step={step} onSelect={(id) => setStep(step === id ? null : id)} />
          <ol className="steps">
            {STEP_ORDER.map((id, i) => {
              const s = turn.steps[id];
              const state = s ? "done" : id === active ? "active" : "pending";
              return (
                <li key={id} className={`step is-${state}`}>
                  <button className="step-head" disabled={!s} onClick={() => setStep(step === id ? null : id)} aria-expanded={step === id}>
                    <span className="step-n">{i + 1}</span>
                    <span className="step-title">{TITLES[id]}</span>
                    <span className="step-sum">{s ? summary(id, s.data) : state === "active" ? "Running…" : ""}</span>
                    <span className="step-ms mono">{s && s.ms ? ms(s.ms) : ""}</span>
                  </button>
                  {s && step === id ? (
                    <div className="detail">
                      <Detail id={id} data={s.data} />
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </>
      ) : null}
    </section>
  );
}
