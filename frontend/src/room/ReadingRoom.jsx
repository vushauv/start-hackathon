import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { askAndWait, getMeta, getPeople } from '../lib/api.js'
import { roomHash } from '../lib/route.js'
import { toQuestionDate } from '../lib/format.js'
import People, { PeopleSelect } from './People.jsx'
import Page from './Page.jsx'
import Ask from './Ask.jsx'
import Transcript from './Transcript.jsx'
import './room.css'

const KEY = 'room.threads.v1'

function loadThreads() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '{}')
  } catch {
    return {}
  }
}

function saveThreads(threads) {
  try {
    const slim = {}
    for (const [k, v] of Object.entries(threads)) slim[k] = v.filter((e) => e.status !== 'pending').slice(-30)
    localStorage.setItem(KEY, JSON.stringify(slim))
  } catch {
    /* storage unavailable: the thread just lives for this visit */
  }
}

export default function ReadingRoom({ route }) {
  const [meta, setMeta] = useState(null)
  const [people, setPeople] = useState(null)
  const [error, setError] = useState(null)
  const [threads, setThreads] = useState(loadThreads)
  const endRef = useRef(null)

  const split = meta?.default_split || 'test'

  useEffect(() => {
    getMeta()
      .then((m) => {
        setMeta(m)
        return getPeople(m.default_split || 'test').then((r) => setPeople(r.people))
      })
      .catch((e) => setError(e.message))
  }, [])

  const hid = route.historyId || people?.[0]?.history_id || null
  const person = people?.find((p) => p.history_id === hid) || null
  const key = `${split}:${hid}`
  const entries = threads[key] || []
  const busy = Object.values(threads).some((t) => t.some((e) => e.status === 'pending'))

  useEffect(() => saveThreads(threads), [threads])

  const patch = useCallback((k, id, change) => {
    setThreads((t) => ({ ...t, [k]: (t[k] || []).map((e) => (e.id === id ? { ...e, ...change } : e)) }))
  }, [])

  const ask = (question, provider, date) => {
    const id = Math.random().toString(36).slice(2, 10)
    const k = key
    const body = { split, history_id: hid, question, provider, question_date: toQuestionDate(date) || undefined }
    setThreads((t) => ({
      ...t,
      [k]: [...(t[k] || []), { id, question, provider, status: 'pending', stage: 'queued', elapsed: 0 }],
    }))
    askAndWait(body, (job) => patch(k, id, { stage: job.stage, elapsed: job.elapsed_s })).then(
      (result) => patch(k, id, { status: 'done', result }),
      (e) => patch(k, id, { status: 'error', error: e.message }),
    )
  }

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' })
  }, [entries.length, entries.at(-1)?.status])

  const go = (h) => {
    location.hash = h
  }
  const open = (c, q) =>
    go(roomHash(hid, { sessionId: c.session_id, turnId: q?.turn_id, start: q?.start, end: q?.end }))

  const quotes = useMemo(
    () => entries.flatMap((e) => (e.result?.citations || []).flatMap((c) => c.quotes)),
    [entries],
  )
  const paneOpen = Boolean(route.sessionId && hid)

  if (error)
    return (
      <div className="room room-error">
        <div className="problem">
          <p>The backend is not answering: {error}</p>
          <p className="mono faint">Start it with: cd pipeline &amp;&amp; uv run pipeline ui --split test</p>
        </div>
      </div>
    )

  return (
    <div className="room" data-pane={paneOpen ? 'open' : 'closed'}>
      <People people={people} current={hid} onPick={(h) => go(roomHash(h))} />
      <section className="page-col">
        <PeopleSelect people={people} current={hid} onPick={(h) => go(roomHash(h))} />
        <Page person={person} entries={entries} onOpen={open} endRef={endRef} />
        <Ask
          meta={meta}
          suggestions={(person?.questions || []).slice(0, 4)}
          showSuggestions={entries.length === 0}
          busy={busy || !person}
          onAsk={ask}
        />
      </section>
      <aside className="pane" aria-hidden={!paneOpen}>
        {paneOpen && (
          <Transcript
            key={route.sessionId}
            split={split}
            historyId={hid}
            route={route}
            quotes={quotes}
            onNavigate={(sid) => go(roomHash(hid, { sessionId: sid }))}
            onClose={() => go(roomHash(hid))}
          />
        )}
      </aside>
    </div>
  )
}
