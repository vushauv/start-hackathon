import { AgentInterface } from '@openuidev/react-ui'
import '@openuidev/react-ui/components.css'
import '@openuidev/react-ui/defaults.css'
import { useEffect, useMemo, useRef, useState } from 'react'
import { getHistory, getMeta, getPeople } from '../lib/api.js'
import { clip, fmtDate, personLabel } from '../lib/format.js'
import { roomHash } from '../lib/route.js'
import Transcript from '../room/Transcript.jsx'
import { makeAssistantMessage } from './Answer.jsx'
import SlashComposer from './SlashComposer.jsx'
import { makeLLM } from './llm.js'
import '../room/room.css'
import './chat.css'

export default function ChatApp({ route }) {
  const [meta, setMeta] = useState(null)
  const [people, setPeople] = useState(null)
  const [error, setError] = useState(null)
  const [options, setOptions] = useState({ provider: null })
  const [quotes, setQuotes] = useState([])
  const [sessions, setSessions] = useState(null)
  const [filter, setFilter] = useState('')
  const optionsRef = useRef(options)
  optionsRef.current = options

  useEffect(() => {
    getMeta()
      .then((m) => {
        setMeta(m)
        return getPeople(m.default_split || 'test').then((r) => setPeople(r.people))
      })
      .catch((e) => setError(e.message))
  }, [])

  const split = meta?.default_split || 'test'
  const hid = route.historyId || people?.[0]?.history_id || null
  const go = (h) => {
    location.hash = h
  }

  useEffect(() => {
    if (!hid) return
    let live = true
    setSessions(null)
    setFilter('')
    getHistory(split, hid).then((r) => live && setSessions(r.sessions), () => live && setSessions([]))
    return () => {
      live = false
    }
  }, [split, hid])

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase()
    if (!f || !sessions) return sessions
    return sessions.filter((x) => (x.date + ' ' + x.preview).toLowerCase().includes(f))
  }, [sessions, filter])

  const llm = useMemo(
    () =>
      hid
        ? makeLLM({
            split,
            historyId: hid,
            getOptions: () => ({ provider: optionsRef.current.provider || undefined }),
            onResult: (r) => setQuotes((q) => [...q, ...(r.citations || []).flatMap((c) => c.quotes)]),
          })
        : null,
    [split, hid],
  )
  const components = useMemo(() => ({ AssistantMessage: makeAssistantMessage(hid) }), [hid])
  const paneOpen = Boolean(route.sessionId && hid)

  if (error)
    return (
      <div className="rc-error-page">
        <p>The backend is not answering: {error}</p>
        <p>Start it with: <code>cd pipeline &amp;&amp; uv run pipeline ui --split test</code></p>
      </div>
    )
  if (!llm) return <div className="rc-error-page"><p>Loading…</p></div>

  return (
    <div className="rc-shell" data-pane={paneOpen ? 'open' : 'closed'}>
      <div className="rc-chat">
        {/* keyed so each person gets a fresh conversation */}
        <AgentInterface
          key={hid}
          llm={llm}
          components={components}
          agentName="Memory"
          scrollVariant="always"
        >
          <AgentInterface.Sidebar>
            <AgentInterface.SidebarContent>
              <div className="rc-side">
                <div className="rc-side-head">
                  <b>Sessions</b>
                  <span>{sessions ? `${shown.length} of ${sessions.length}` : '…'}</span>
                </div>
                <input
                  className="rc-filter"
                  type="search"
                  placeholder="Filter by date or text"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  aria-label="Filter sessions"
                />
                <ul className="rc-sessions">
                  {!sessions && <li className="rc-hint">Loading…</li>}
                  {shown?.map((x) => (
                    <li key={x.session_id}>
                      <a
                        href={roomHash(hid, { sessionId: x.session_id })}
                        aria-current={x.session_id === route.sessionId ? 'true' : undefined}
                      >
                        <span className="rc-date">{fmtDate(x.date)}</span>
                        <span className="rc-prev">{clip(x.preview, 90)}</span>
                      </a>
                    </li>
                  ))}
                  {shown?.length === 0 && <li className="rc-hint">No session matches.</li>}
                </ul>
              </div>
            </AgentInterface.SidebarContent>
            <AgentInterface.SidebarFooter>
              <div className="rc-who">
                <label htmlFor="rc-person">Person</label>
                <select id="rc-person" value={hid} onChange={(e) => go(roomHash(e.target.value))}>
                  {people?.map((p) => (
                    <option key={p.history_id} value={p.history_id}>
                      {personLabel(p)}
                    </option>
                  ))}
                </select>
                <a href="#/explore">dataset explorer</a>
              </div>
            </AgentInterface.SidebarFooter>
          </AgentInterface.Sidebar>
          <AgentInterface.Composer>
            <SlashComposer
              meta={meta}
              options={options}
              setOptions={setOptions}
              disabled={!hid}
              person={people?.find((p) => p.history_id === hid)?.name}
              starters={(people?.find((p) => p.history_id === hid)?.questions || []).slice(0, 4).map((q) => q.question)}
            />
          </AgentInterface.Composer>
        </AgentInterface>
      </div>
      <aside className="rc-pane" aria-hidden={!paneOpen}>
        {paneOpen && (
          <div className="room rc-scope">
          <Transcript
            key={route.sessionId}
            split={split}
            historyId={hid}
            route={route}
            quotes={quotes}
            onNavigate={(sid) => go(roomHash(hid, { sessionId: sid }))}
            onClose={() => go(roomHash(hid))}
          />
          </div>
        )}
      </aside>
    </div>
  )
}
