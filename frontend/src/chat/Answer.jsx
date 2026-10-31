import { clip, fmtDate } from '../lib/format.js'
import { roomHash } from '../lib/route.js'
import { RESULT_MARK, STAGE_MARK } from './llm.js'

const STAGE = {
  queued: 'Waiting for the agent',
  retrieving: 'Searching the sessions',
  answering: 'Reading what it found',
  locating: 'Marking the passages',
}

export function parseContent(text) {
  const i = (text || '').lastIndexOf(RESULT_MARK)
  if (i >= 0) {
    try {
      return { result: JSON.parse(text.slice(i + 1)) }
    } catch {
      return { stage: 'Finishing' }
    }
  }
  const stages = (text || '').split(STAGE_MARK).filter(Boolean)
  const s = stages.at(-1)?.trim()
  return { stage: STAGE[s] || s || 'Working' }
}

function Hint({ r }) {
  return (
    <p className="rc-quiet">
      {r.unknown
        ? `There is no /${r.unknown} tool. `
        : r.empty
          ? 'Add a question after the tool name. '
          : 'Type / to pick a tool. '}
      Try <code>/recall what did they say about …</code> to search this person’s conversations.
    </p>
  )
}

export function makeAssistantMessage(historyId) {
  return function RecallMessage({ message, isStreaming }) {
    const { result: r, stage } = parseContent(message.content)
    if (!r)
      return (
        <div className="rc-msg" role="status" aria-busy={isStreaming}>
          <p className="rc-stage">{stage}…</p>
        </div>
      )
    if (r.kind === 'error') return <div className="rc-msg"><p className="rc-error">{r.message}</p></div>
    if (r.kind === 'hint') return <div className="rc-msg"><Hint r={r} /></div>

    const cites = r.citations || []
    const link = (c, q) =>
      roomHash(historyId, { sessionId: c.session_id, turnId: q?.turn_id, start: q?.start, end: q?.end })
    return (
      <div className="rc-msg">
        {r.abstained ? (
          <p className="rc-quiet">Nothing in these conversations settles this.</p>
        ) : (
          <p className="rc-answer">{r.answer || '(empty answer)'}</p>
        )}
        {cites.length > 0 && (
          <ol className="rc-sources" aria-label="Sources">
            {cites.map((c, i) => (
              <li key={c.session_id}>
                <a className="rc-session" href={link(c, c.quotes[0])}>
                  <span className="rc-n">{i + 1}</span> Session of {fmtDate(c.date)}
                </a>
                {c.quotes.map((q) => (
                  <a key={q.turn_id + q.start} className="rc-quote" href={link(c, q)}>
                    “{clip(q.text, 160)}”
                  </a>
                ))}
              </li>
            ))}
          </ol>
        )}
        {cites.length === 0 && r.abstained && r.sent?.length > 0 && (
          <p className="rc-looked">
            Looked at{' '}
            {r.sent.map((s, i) => (
              <span key={s.session_id}>
                {i > 0 && ', '}
                <a href={roomHash(historyId, { sessionId: s.session_id })}>{fmtDate(s.date)}</a>
              </span>
            ))}
          </p>
        )}
        <p className="rc-meta">
          {r.provider}, {r.latency_s ? `${r.latency_s.toFixed(0)} s` : 'instant'}, as of {fmtDate(r.question_date)}
        </p>
      </div>
    )
  }
}
