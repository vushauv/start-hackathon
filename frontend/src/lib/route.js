// Hash routes: "#/" | "#/h/<history_id>?s=<session_id>&t=<turn_id>&q=<quote index>" | "#/explore"
export function parseHash(hash) {
  const raw = (hash || '').replace(/^#/, '') || '/'
  const [path, query = ''] = raw.split('?')
  const params = new URLSearchParams(query)
  if (path.startsWith('/explore')) return { page: 'explore' }
  const m = /^\/h\/([^/]+)/.exec(path)
  return {
    page: 'room',
    historyId: m ? decodeURIComponent(m[1]) : null,
    sessionId: params.get('s'),
    turnId: params.get('t'),
    start: params.get('a') == null ? null : Number(params.get('a')),
    end: params.get('b') == null ? null : Number(params.get('b')),
  }
}

export function roomHash(historyId, { sessionId, turnId, start, end } = {}) {
  if (!historyId) return '#/'
  const p = new URLSearchParams()
  if (sessionId) p.set('s', sessionId)
  if (turnId) p.set('t', turnId)
  if (start != null) p.set('a', String(start))
  if (end != null) p.set('b', String(end))
  const q = p.toString()
  return `#/h/${encodeURIComponent(historyId)}${q ? '?' + q : ''}`
}
