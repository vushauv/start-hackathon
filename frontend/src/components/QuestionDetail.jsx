import { useEffect, useState } from 'react'
import { loadByKey } from '../lib/jsonl.js'

const LIMIT = 1500

function Turn({ turn }) {
  const [full, setFull] = useState(false)
  const long = turn.content.length > LIMIT
  const text = full || !long ? turn.content : turn.content.slice(0, LIMIT) + '…'
  return (
    <div className={`turn ${turn.role}`}>
      <b>{turn.role}</b>
      <div className="text">{text}</div>
      {long && (
        <button className="link" onClick={() => setFull(!full)}>
          {full ? 'show less' : `show all (${turn.content.length.toLocaleString()} chars)`}
        </button>
      )}
    </div>
  )
}

function Session({ session, evidence, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className={`session ${evidence ? 'evidence' : ''}`}>
      <header onClick={() => setOpen(!open)}>
        <span>{open ? '▾' : '▸'}</span>
        <span className="date">{session.date}</span>
        <code>{session.session_id}</code>
        <span className="muted">{session.turns.length} turns</span>
        {evidence && <span className="tag evidence">evidence</span>}
      </header>
      {open && session.turns.map((t) => <Turn key={t.turn_id} turn={t} />)}
    </section>
  )
}

export default function QuestionDetail({ question, gold }) {
  const [history, setHistory] = useState(null)
  const [error, setError] = useState(null)
  const [onlyEvidence, setOnlyEvidence] = useState(false)

  useEffect(() => {
    let live = true
    loadByKey('histories', question.history_id).then(
      (h) => live && setHistory(h),
      (e) => live && setError(e),
    )
    return () => {
      live = false
    }
  }, [question.history_id])

  const ev = new Set(gold.evidence)
  const sessions = history?.sessions.filter((s) => !onlyEvidence || ev.has(s.session_id))

  return (
    <div className="detail">
      <h2>{question.question}</h2>
      <dl>
        <dt>Type</dt>
        <dd>{gold.question_type}</dd>
        <dt>Asked</dt>
        <dd>{question.question_date}</dd>
        <dt>History</dt>
        <dd>
          <code>{question.history_id}</code>
        </dd>
        <dt>Gold answer</dt>
        <dd className="answer">{gold.abstain ? <i>Abstain – not answerable ({gold.answer})</i> : gold.answer}</dd>
        <dt>Evidence</dt>
        <dd>{gold.evidence.length ? gold.evidence.map((e) => <code key={e}>{e} </code>) : <span className="muted">none</span>}</dd>
      </dl>

      <div className="bar">
        <h3>Sessions {history && `(${history.sessions.length})`}</h3>
        <label>
          <input type="checkbox" checked={onlyEvidence} onChange={(e) => setOnlyEvidence(e.target.checked)} /> Evidence only
        </label>
      </div>
      {error && <p className="error">{String(error.message || error)}</p>}
      {!history && !error && <p className="muted">Loading history…</p>}
      {sessions?.map((s) => (
        <Session key={s.session_id} session={s} evidence={ev.has(s.session_id)} defaultOpen={ev.has(s.session_id)} />
      ))}
    </div>
  )
}
