const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

// "2023/05/28 (Sun) 11:55" -> {y, m, d} or null
export function parseDate(s) {
  const m = /(\d{4})[/-](\d{2})[/-](\d{2})/.exec(s || '')
  return m ? { y: +m[1], m: +m[2], d: +m[3] } : null
}

export function fmtDate(s, withYear = true) {
  const p = parseDate(s)
  if (!p) return s || ''
  return `${p.d} ${MONTHS[p.m - 1].slice(0, 3)}${withYear ? ' ' + p.y : ''}`
}

export function fmtSpan(a, b) {
  const x = parseDate(a)
  const y = parseDate(b)
  if (!x || !y) return ''
  if (x.y === y.y && x.m === y.m) return `${MONTHS[x.m - 1]} ${x.y}`
  if (x.y === y.y) return `${MONTHS[x.m - 1]} to ${MONTHS[y.m - 1]} ${y.y}`
  return `${MONTHS[x.m - 1]} ${x.y} to ${MONTHS[y.m - 1]} ${y.y}`
}

// <input type="date"> value "2023-09-14" -> the dataset's question_date format
export function toQuestionDate(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || '')
  if (!m) return ''
  const dt = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]))
  return `${m[1]}/${m[2]}/${m[3]} (${DAYS[dt.getUTCDay()]}) 23:59`
}

export function personLabel(p) {
  return p.name || `Person ${p.history_id.replace(/^h_/, '').slice(0, 4)}`
}

export function clip(s, n) {
  const t = (s || '').replace(/\s+/g, ' ').trim()
  return t.length > n ? t.slice(0, n - 1).trimEnd() + '…' : t
}

// The current moment in the dataset's question_date format, e.g. "2026/10/31 (Sat) 14:05"
export function nowQuestionDate(d = new Date()) {
  const z = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}/${z(d.getMonth() + 1)}/${z(d.getDate())} (${DAYS[d.getDay()]}) ${z(d.getHours())}:${z(d.getMinutes())}`
}
