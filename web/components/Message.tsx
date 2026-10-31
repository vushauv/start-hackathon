"use client";

import { Fragment, useEffect, useState } from "react";
import type { Turn } from "@/lib/types";
import { Trace } from "./Trace";
import { ms, shortDate } from "./format";

const STATUS: Record<string, { label: string; tone: string }> = {
  answered: { label: "Answered", tone: "ok" },
  abstained: { label: "Abstained", tone: "warn" },
  "no-answer": { label: "No answer", tone: "err" },
  "dry-run": { label: "Dry run", tone: "" },
};

/** Render "[E1]" markers as citation labels. */
function WithCitations({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\[E\d+\])/g).map((p, i) =>
        /^\[E\d+\]$/.test(p) ? (
          <sup key={i} className="cite">
            {p.slice(1, -1)}
          </sup>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        ),
      )}
    </>
  );
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, []);
  return <span className="mono">{((now - since) / 1000).toFixed(1)} s</span>;
}

export function Message({ turn, isLast }: { turn: Turn; isLast: boolean }) {
  const [shown, setShown] = useState<"reasoning" | "reply" | null>(null);
  const a = turn.answer;
  const st = a ? STATUS[a.status] : null;
  const sent = new Set(turn.steps.evidence?.data.passages.map((p) => p.id) ?? []);
  const live = turn.phase === "queued" || turn.phase === "retrieving" || turn.phase === "waiting";
  const hasReply = !!a?.reply && a.reply !== a.answer;
  const toggle = (k: "reasoning" | "reply") => setShown(shown === k ? null : k);

  return (
    <article className="turn">
      <div className="question">
        <p>{turn.question}</p>
        {turn.questionId ? <span className="mono">{turn.questionId}</span> : null}
      </div>

      {live ? (
        <div className="status-line">
          <span className="dot live" />
          <strong>
            {turn.phase === "queued" ? "Queued" : turn.phase === "retrieving" ? "Retrieving" : `Waiting for ${turn.waiting?.who}`}
          </strong>
          {turn.phase === "waiting" && turn.waiting ? <span>{turn.waiting.expect}</span> : null}
          <span className="push">
            <Elapsed since={turn.startedAt} />
          </span>
        </div>
      ) : null}

      {turn.phase === "error" ? (
        <>
          <div className="status-line">
            <span className="dot err" />
            <strong>Error</strong>
          </div>
          <p className="error-text mono">{turn.error}</p>
        </>
      ) : null}

      {a ? (
        <>
          <div className="status-line">
            <span className={`dot ${st?.tone}`} />
            <strong>{st?.label}</strong>
            <span>{a.who}</span>
            {a.latency_s ? <span>· {a.latency_s.toFixed(1)} s</span> : null}
            {a.attempts && a.attempts > 1 ? <span>· attempt {a.attempts}</span> : null}
            {turn.totalMs ? <span className="push mono">{ms(turn.totalMs)}</span> : null}
          </div>
          {a.status === "dry-run" ? (
            <p className="answer is-note">Retrieval only. Turn off dry run in settings to get an answer.</p>
          ) : a.status === "no-answer" ? (
            <p className="answer is-note">{a.error || "No answer after retries."}</p>
          ) : (
            <p className="answer">
              <WithCitations text={a.answer || "(empty reply)"} />
            </p>
          )}
        </>
      ) : null}

      {a?.citations.length || turn.gold ? (
        <dl className="facts">
          {a?.citations.length ? (
            <>
              <dt>Sources</dt>
              <dd>
                {a.citations.map((c) => (
                  <span key={c.label + c.id} className="tag" title={shortDate(c.date)}>
                    <span className="cite">{c.label}</span>&nbsp;{c.id}
                  </span>
                ))}
              </dd>
            </>
          ) : null}
          {turn.gold ? (
            <>
              <dt>Expected</dt>
              <dd>{turn.gold.abstain ? "Abstain (unanswerable)" : turn.gold.answer}</dd>
              {turn.gold.evidence.length ? (
                <>
                  <dt>Evidence</dt>
                  <dd>
                    {turn.gold.evidence.map((e) => (
                      <span key={e} className={`tag ${sent.has(e) ? "tag-hit" : ""}`} title={sent.has(e) ? "Retrieved" : "Missed"}>
                        {e}
                      </span>
                    ))}
                    <span className="muted">
                      {turn.gold.evidence.filter((e) => sent.has(e)).length} of {turn.gold.evidence.length} retrieved
                    </span>
                  </dd>
                </>
              ) : null}
            </>
          ) : null}
        </dl>
      ) : null}

      {a?.reasoning || hasReply ? (
        <>
          <div className="links">
            {a?.reasoning ? (
              <button className="link" aria-expanded={shown === "reasoning"} onClick={() => toggle("reasoning")}>
                Reasoning
              </button>
            ) : null}
            {hasReply ? (
              <button className="link" aria-expanded={shown === "reply"} onClick={() => toggle("reply")}>
                Full reply
              </button>
            ) : null}
          </div>
          {shown ? <pre className="pre">{shown === "reasoning" ? a?.reasoning : a?.reply}</pre> : null}
        </>
      ) : null}

      {Object.keys(turn.steps).length ? <Trace turn={turn} defaultOpen={isLast && live} /> : null}
    </article>
  );
}
