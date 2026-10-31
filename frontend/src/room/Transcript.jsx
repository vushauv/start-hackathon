import { useEffect, useRef, useState } from 'react'
import { getSession } from '../lib/api.js'
import { fmtDate } from '../lib/format.js'

function segments(content, marks) {
  const out = []
  let i = 0
  for (const m of [...marks].sort((a, b) => a.start - b.start)) {
    if (m.start < i || m.end > content.length || m.end <= m.start) continue
    if (m.start > i) out.push({ t: content.slice(i, m.start) })
    out.push({ t: content.slice(m.start, m.end), mark: m })
    i = m.end
  }
  if (i < content.length) out.push({ t: content.slice(i) })
  return out
}

export default function Transcript({ split, historyId, route, quotes, onNavigate, onClose }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const body = useRef(null)
  const { sessionId, turnId } = route

  useEffect(() => {
    let live = true
    setData(null)
    setError(null)
    getSession(split, historyId, sessionId).then(
      (d) => live && setData(d),
      (e) => live && setError(e.message),
    )
    return () => {
      live = false
    }
  }, [split, historyId, sessionId])

  // marks for this session: every quote of the thread, plus the one in the URL (a shared link)
  const marks = {}
  const add = (m) => {
    const list = (marks[m.turn_id] ||= [])
    if (!list.some((x) => x.start === m.start)) list.push(m)
  }
  quotes.filter((q) => q.session_id === sessionId).forEach(add)
  if (turnId && route.start != null && route.end != null) add({ turn_id: turnId, start: route.start, end: route.end })
  const isActive = (m) => m.turn_id === turnId && (route.start == null || m.start === route.start)

  useEffect(() => {
    if (!data || !body.current) return
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const el = turnId ? body.current.querySelector(`[id="${CSS.escape(turnId)}"]`) : null
    if (el) el.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' })
    else body.current.scrollTo({ top: 0 })
  }, [data, turnId, route.start])

  return (
    <div className="pane-inner">
      <header className="pane-head">
        <div>
          <p className="pane-date">{data ? fmtDate(data.date) : '…'}</p>
          <p className="mono faint">{data ? `session ${data.index + 1} of ${data.total}` : sessionId}</p>
        </div>
        <div className="pane-nav">
          <button disabled={!data?.prev} onClick={() => onNavigate(data.prev)} aria-label="Previous session">
            earlier
          </button>
          <button disabled={!data?.next} onClick={() => onNavigate(data.next)} aria-label="Next session">
            later
          </button>
          <button onClick={onClose} aria-label="Close transcript">
            close
          </button>
        </div>
      </header>
      <div className="pane-body" ref={body}>
        {error && <p className="problem">{error}</p>}
        {!data && !error && <p className="faint">Loading…</p>}
        {data?.turns.map((t) => (
          <div key={t.turn_id} id={t.turn_id} className={`turn ${t.role}`} data-target={t.turn_id === turnId || undefined}>
            <span className="role mono">{t.role}</span>
            <p>
              {segments(t.content || '', marks[t.turn_id] || []).map((s, i) =>
                s.mark ? (
                  <mark key={i} className="quote" data-active={isActive(s.mark) || undefined}>
                    {s.t}
                  </mark>
                ) : (
                  <span key={i}>{s.t}</span>
                ),
              )}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
