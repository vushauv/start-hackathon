import { useThread } from '@openuidev/react-headless'
import { useRef, useState } from 'react'
import { TOOLS } from './llm.js'

export default function SlashComposer({ meta, options, setOptions, disabled, person, starters = [] }) {
  const { messages, processMessage, isRunning, cancelMessage } = useThread()
  const empty = !messages || messages.length === 0
  const [text, setText] = useState('')
  const [active, setActive] = useState(0)
  const ta = useRef(null)

  const typing = /^\/([a-z-]*)$/i.exec(text)
  const matches = typing ? TOOLS.filter((t) => t.name.startsWith(typing[1].toLowerCase())) : []
  const menu = matches.length > 0

  const pick = (t) => {
    setText(`/${t.name} `)
    ta.current?.focus()
  }
  const send = () => {
    const v = text.trim()
    if (!v || isRunning || disabled) return
    processMessage({ role: 'user', content: v })
    setText('')
    if (ta.current) ta.current.style.height = ''
  }

  return (
    <>
    {empty && (
      <div className="rc-empty">
        <h2>{person ? `Ask about ${person}` : 'Ask about this person'}</h2>
        <p>
          Type <code>/recall</code> followed by a question. It searches this person’s conversations, answers, and
          links the exact quotes it used.
        </p>
        <ul>
          {(starters).map((q) => (
            <li key={q}>
              <button type="button" disabled={isRunning || disabled} onClick={() => processMessage({ role: 'user', content: `/recall ${q}` })}>
                {q}
              </button>
            </li>
          ))}
        </ul>
      </div>
    )}
    <div className="rc-composer">
      {menu && (
        <ul className="rc-menu" role="listbox" aria-label="Tools">
          {matches.map((t, i) => (
            <li key={t.name} role="option" aria-selected={i === active}>
              <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => pick(t)}>
                <code>/{t.name}</code>
                <span>{t.hint}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <textarea
        ref={ta}
        rows={1}
        value={text}
        aria-label="Message"
        placeholder="Type / for tools, e.g. /recall where did they travel last summer"
        onChange={(e) => {
          setText(e.target.value)
          setActive(0)
          e.target.style.height = 'auto'
          e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
        }}
        onKeyDown={(e) => {
          if (menu && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
            e.preventDefault()
            setActive((a) => (a + (e.key === 'ArrowDown' ? 1 : -1) + matches.length) % matches.length)
          } else if (menu && (e.key === 'Tab' || e.key === 'Enter')) {
            e.preventDefault()
            pick(matches[active] || matches[0])
          } else if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            send()
          }
        }}
      />
      <div className="rc-row">
        {meta?.providers && !meta.dry_run && (
          <span className="rc-seg" role="group" aria-label="Who answers">
            {meta.providers.map((p) => (
              <button
                type="button"
                key={p}
                aria-pressed={p === (options.provider || meta.provider)}
                onClick={() => setOptions((o) => ({ ...o, provider: p }))}
              >
                {p}
              </button>
            ))}
          </span>
        )}
        {meta?.dry_run && <span>dry run</span>}
        {isRunning ? (
          <button type="button" className="rc-send" onClick={cancelMessage}>Stop</button>
        ) : (
          <button type="button" className="rc-send" onClick={send} disabled={!text.trim() || disabled}>Send</button>
        )}
      </div>
    </div>
    </>
  )
}
