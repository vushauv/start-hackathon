import { useRef, useState } from 'react'

export default function Ask({ meta, suggestions, showSuggestions, busy, onAsk }) {
  const [text, setText] = useState('')
  const [date, setDate] = useState('')
  const [provider, setProvider] = useState(null)
  const ta = useRef(null)
  const prov = provider || meta?.provider || 'reactor'

  const submit = (q) => {
    const question = (q ?? text).trim()
    if (!question || busy) return
    onAsk(question, prov, date)
    setText('')
    if (ta.current) ta.current.style.height = ''
  }

  return (
    <div className="ask">
      {showSuggestions && suggestions.length > 0 && (
        <ul className="chips" aria-label="Suggested questions">
          {suggestions.map((s) => (
            <li key={s.question_id}>
              <button className="chip" disabled={busy} onClick={() => submit(s.question)}>
                {s.question}
              </button>
            </li>
          ))}
        </ul>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <textarea
          ref={ta}
          rows={1}
          value={text}
          placeholder="Ask about this person"
          aria-label="Question"
          onChange={(e) => {
            setText(e.target.value)
            e.target.style.height = 'auto'
            e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        <div className="ask-row">
          <label className="mono faint">
            as of{' '}
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)} aria-label="Question date" />
          </label>
          {meta?.providers && !meta.dry_run && (
            <span className="seg mono" role="group" aria-label="Who answers">
              {meta.providers.map((p) => (
                <button type="button" key={p} aria-pressed={p === prov} onClick={() => setProvider(p)}>
                  {p}
                </button>
              ))}
            </span>
          )}
          {meta?.dry_run && <span className="mono faint">dry run</span>}
          <button className="send" type="submit" disabled={busy || !text.trim()}>
            Ask
          </button>
        </div>
      </form>
    </div>
  )
}
