// Lazy JSONL access. The dev server exposes /data/<name>.jsonl (Range-capable)
// and /data/<name>.jsonl.index (byte offsets + ids), so records are fetched
// on demand instead of downloading whole files.

const CHUNK = 100
const indexes = new Map()
const stores = new Map()

export function getIndex(name) {
  if (!indexes.has(name)) {
    const p = fetch(`/data/${name}.jsonl.index`).then(async (r) => {
      if (!r.ok) throw new Error(await r.text())
      return r.json()
    })
    p.catch(() => indexes.delete(name))
    indexes.set(name, p)
  }
  return indexes.get(name)
}

// Reads lines [from, to) with a single Range request.
export async function readLines(name, from, to) {
  const { offsets, lengths } = await getIndex(name)
  if (to <= from) return []
  const start = offsets[from]
  const end = offsets[to - 1] + lengths[to - 1] - 1
  const res = await fetch(`/data/${name}.jsonl`, { headers: { Range: `bytes=${start}-${end}` } })
  if (!res.ok) throw new Error(`Failed to read ${name}: ${res.status}`)
  return (await res.text())
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l))
}

// Loads records sequentially in chunks until at least `wanted` are cached.
export function ensureLoaded(name, wanted) {
  if (!stores.has(name)) stores.set(name, { rows: [], chain: Promise.resolve() })
  const s = stores.get(name)
  const p = s.chain.then(async () => {
    const { ids } = await getIndex(name)
    const target = Math.min(wanted, ids.length)
    while (s.rows.length < target) {
      const from = s.rows.length
      s.rows.push(...(await readLines(name, from, Math.min(from + CHUNK, ids.length))))
    }
    return { rows: s.rows.slice(), total: ids.length }
  })
  s.chain = p.catch(() => {})
  return p
}

const histories = new Map()

export async function loadByKey(name, id) {
  const key = `${name}:${id}`
  if (!histories.has(key)) {
    const p = getIndex(name).then(async ({ ids }) => {
      const i = ids.indexOf(id)
      if (i < 0) throw new Error(`${id} not found in ${name}`)
      return (await readLines(name, i, i + 1))[0]
    })
    p.catch(() => histories.delete(key))
    histories.set(key, p)
    if (histories.size > 8) histories.delete(histories.keys().next().value)
  }
  return histories.get(key)
}
