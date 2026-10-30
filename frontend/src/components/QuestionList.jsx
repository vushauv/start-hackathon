import { useMemo, useState } from 'react'
import { useRecords } from '../lib/useRecords.js'

const PAGE = 50
const TYPES = [
  'multi-session',
  'temporal-reasoning',
  'single-session-assistant',
  'knowledge-update',
  'single-session-user',
]

export default function QuestionList({ selectedId, onSelect }) {
  const [page, setPage] = useState(0)
  const [type, setType] = useState('')
  const [abstain, setAbstain] = useState(false)
  const [search, setSearch] = useState('')

  const filtering = Boolean(type || abstain || search)
  const wanted = filtering ? Infinity : (page + 1) * PAGE
  const q = useRecords('questions', wanted)
  const g = useRecords('gold', wanted)

  const items = useMemo(() => {
    const n = Math.min(q.rows.length, g.rows.length)
    const needle = search.toLowerCase()
    const out = []
    for (let i = 0; i < n; i++) {
      const gold = g.rows[i]
      const question = q.rows[i]
      if (type && gold.question_type !== type) continue
      if (abstain && !gold.abstain) continue
      if (needle && !question.question.toLowerCase().includes(needle)) continue
      out.push({ question, gold })
    }
    return out
  }, [q.rows, g.rows, type, abstain, search])

  const total = q.total ?? 0
  const pages = Math.max(1, Math.ceil((filtering ? items.length : total) / PAGE))
  const cur = Math.min(page, pages - 1)
  const visible = items.slice(cur * PAGE, (cur + 1) * PAGE)
  const loading = q.loading || g.loading
  const error = q.error || g.error
  const reset = (fn) => (v) => {
    fn(v)
    setPage(0)
  }

  return (
    <div className="list">
      <div className="filters">
        <input placeholder="Search questions…" value={search} onChange={(e) => reset(setSearch)(e.target.value)} />
        <select value={type} onChange={(e) => reset(setType)(e.target.value)}>
          <option value="">All types</option>
          {TYPES.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <label>
          <input type="checkbox" checked={abstain} onChange={(e) => reset(setAbstain)(e.target.checked)} /> Abstain only
        </label>
      </div>
      {error && <p className="error">{String(error.message || error)}</p>}
      <ul>
        {visible.map(({ question, gold }) => (
          <li
            key={question.question_id}
            className={question.question_id === selectedId ? 'active' : ''}
            onClick={() => onSelect({ question, gold })}
          >
            <span className={`tag t-${gold.question_type}`}>{gold.question_type}</span>
            {gold.abstain && <span className="tag abstain">abstain</span>}
            <div>{question.question}</div>
          </li>
        ))}
      </ul>
      {loading && <p className="muted pad">Loading…</p>}
      <div className="pager">
        <button disabled={cur === 0} onClick={() => setPage(cur - 1)}>
          ←
        </button>
        <span>
          {cur + 1} / {pages}
          {filtering ? ` · ${items.length} match` : total ? ` · ${total} total` : ''}
        </span>
        <button disabled={cur + 1 >= pages} onClick={() => setPage(cur + 1)}>
          →
        </button>
      </div>
    </div>
  )
}
