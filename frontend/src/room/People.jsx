import { fmtSpan, personLabel } from '../lib/format.js'

export default function People({ people, current, onPick }) {
  return (
    <nav className="rail" aria-label="People">
      <h1 className="brand">Reading room</h1>
      <p className="rail-note">Pick a person, then ask about what they have told the assistant.</p>
      {!people && <p className="faint">Loading…</p>}
      <ul className="people">
        {people?.map((p) => (
          <li key={p.history_id}>
            <button
              className="person"
              aria-current={p.history_id === current ? 'true' : undefined}
              onClick={() => onPick(p.history_id)}
            >
              <span className="person-name">{personLabel(p)}</span>
              <span className="mono faint">
                {p.n_sessions} sessions, {fmtSpan(p.first_date, p.last_date)}
              </span>
            </button>
          </li>
        ))}
      </ul>
      <a className="rail-link mono" href="#/explore">dataset explorer</a>
    </nav>
  )
}

export function PeopleSelect({ people, current, onPick }) {
  return (
    <select className="people-select" value={current || ''} onChange={(e) => onPick(e.target.value)} aria-label="Person">
      {people?.map((p) => (
        <option key={p.history_id} value={p.history_id}>
          {personLabel(p)} ({p.n_sessions} sessions)
        </option>
      ))}
    </select>
  )
}
