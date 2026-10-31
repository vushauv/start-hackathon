// Lazy JSONL access through the Vite data middleware and HTTP Range requests.
import type { DatasetName, DatasetRecords, JsonlIndex } from '../types'

const CHUNK = 100
const indexes = new Map<DatasetName, Promise<JsonlIndex>>()

export interface LoadedRecords<T> {
  rows: T[]
  total: number
}

interface RecordStore<T> {
  rows: T[]
  chain: Promise<unknown>
  records: Map<string, Promise<T>>
}

function createStore<T>(): RecordStore<T> {
  return { rows: [], chain: Promise.resolve(), records: new Map() }
}

const stores: { [N in DatasetName]: RecordStore<DatasetRecords[N]> } = {
  questions: createStore(),
  gold: createStore(),
  histories: createStore(),
}

export function getIndex(name: DatasetName): Promise<JsonlIndex> {
  let pending = indexes.get(name)
  if (!pending) {
    pending = fetch(`/data/${name}.jsonl.index`).then(async (r) => {
      if (!r.ok) throw new Error(await r.text())
      return r.json() as Promise<JsonlIndex>
    })
    pending.catch(() => indexes.delete(name))
    indexes.set(name, pending)
  }
  return pending
}

// Reads lines [from, to) with a single Range request.
export async function readLines<N extends DatasetName>(
  name: N,
  from: number,
  to: number,
): Promise<DatasetRecords[N][]> {
  const { offsets, lengths } = await getIndex(name)
  if (to <= from) return []
  const start = offsets[from]
  const end = offsets[to - 1] + lengths[to - 1] - 1
  const res = await fetch(`/data/${name}.jsonl`, { headers: { Range: `bytes=${start}-${end}` } })
  if (!res.ok) throw new Error(`Failed to read ${name}: ${res.status}`)
  return (await res.text())
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as DatasetRecords[N])
}

// Loads records sequentially in chunks until at least `wanted` are cached.
export function ensureLoaded<N extends DatasetName>(
  name: N,
  wanted: number,
): Promise<LoadedRecords<DatasetRecords[N]>> {
  const store: RecordStore<DatasetRecords[N]> = stores[name]
  const pending = store.chain.then(async () => {
    const { ids } = await getIndex(name)
    const target = Math.min(wanted, ids.length)
    while (store.rows.length < target) {
      const from = store.rows.length
      store.rows.push(...(await readLines(name, from, Math.min(from + CHUNK, ids.length))))
    }
    return { rows: store.rows.slice(), total: ids.length }
  })
  store.chain = pending.catch(() => {})
  return pending
}

export async function loadByKey<N extends DatasetName>(name: N, id: string): Promise<DatasetRecords[N]> {
  const store: RecordStore<DatasetRecords[N]> = stores[name]
  let pending = store.records.get(id)
  if (!pending) {
    pending = getIndex(name).then(async ({ ids }) => {
      const i = ids.indexOf(id)
      if (i < 0) throw new Error(`${id} not found in ${name}`)
      return (await readLines(name, i, i + 1))[0]
    })
    pending.catch(() => store.records.delete(id))
    store.records.set(id, pending)
    if (store.records.size > 8) {
      const oldest = store.records.keys().next()
      if (!oldest.done) store.records.delete(oldest.value)
    }
  }
  return pending
}
