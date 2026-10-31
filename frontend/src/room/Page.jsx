import { clip, fmtDate, fmtSpan, personLabel } from '../lib/format.js'

const STAGE = {
  queued: 'waiting for the agent',
  retrieving: 'searching the sessions',
  answering: 'reading what it found',
  locating: 'marking the passages',
}

function stageText(e) {
  if (e.stage === 'answering') return e.provider === 'groq' ? 'asking Groq' : 'asking Reactor'
  return STAGE[e.stage] || e.stage
}

function Leaf({ entry, onOpen }) {
  const r = entry.result
  const cites = r?.citations || []
  return (
    <article className="leaf" aria-busy={entry.status === 'pending'}>
      <p className="question">{entry.question}</p>
      <div className="leaf-body">
        <div className="answer-col">
          {entry.status === 'pending' && (
            <p className="progress mono" role="status">
              {stageText(entry)}
              <span className="faint"> {Math.round(entry.elapsed || 0)}s</span>
            </p>
          )}
          {entry.status === 'error' && <p className="problem">{entry.error}</p>}
          {entry.status === 'done' && !r.abstained && (
            <p className="answer">
              {r.answer || '(empty answer)'}
              {cites.map((c, i) => (
                <sup key={c.session_id}>
                  <button
                    className="mark"
                    aria-label={`Open the session of ${fmtDate(c.date)} at the quoted sentence`}
                    onClick={() => onOpen(c, c.quotes[0])}
                  >
                    {i + 1}
                  </button>
                </sup>
              ))}
            </p>
          )}
          {entry.status === 'done' && r.abstained && (
            <>
              <p className="answer quiet">Nothing in these conversations settles this.</p>
              {cites.length === 0 && r.sent?.length > 0 && (
                <p className="checked">
                  <span className="faint">Looked at </span>
                  {r.sent.map((s, i) => (
                    <span key={s.session_id}>
                      {i > 0 && ', '}
                      <button className="link" onClick={() => onOpen({ session_id: s.session_id }, null)}>
                        {fmtDate(s.date)}
                      </button>
                    </span>
                  ))}
                </p>
              )}
            </>
          )}
        </div>
        {entry.status === 'done' && cites.length > 0 && (
          <ol className="notes">
            {cites.map((c, i) => (
              <li key={c.session_id}>
                <span className="note-n mono">{i + 1}</span>
                <div>
                  <span className="note-date mono">{fmtDate(c.date)}</span>
                  {c.quotes.map((q) => (
                    <button key={q.turn_id + q.start} className="note-quote" onClick={() => onOpen(c, q)}>
                      “{clip(q.text, 150)}”
                    </button>
                  ))}
                  {c.quotes.length === 0 && (
                    <button className="link" onClick={() => onOpen(c, null)}>open session</button>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>
      {entry.status === 'done' && (
        <p className="meta mono faint">
          {r.provider}, {r.latency_s ? `${r.latency_s.toFixed(0)} s` : 'instant'}, as of {fmtDate(r.question_date)}
        </p>
      )}
    </article>
  )
}

export default function Page({ person, entries, onOpen, endRef }) {
  return (
    <div className="page">
      <header className="masthead">
        <h2>{person ? personLabel(person) : '…'}</h2>
        {person && (
          <p className="mono faint">
            {person.n_sessions} sessions, {fmtSpan(person.first_date, person.last_date)}
          </p>
        )}
      </header>
      {entries.length === 0 && (
        <p className="empty">Ask anything this person might have mentioned. Answers come with the sentences they rest on.</p>
      )}
      {entries.map((e) => (
        <Leaf key={e.id} entry={e} onOpen={onOpen} />
      ))}
      <div ref={endRef} />
    </div>
  )
}
