import { useEffect, useState } from 'react'
import { ensureLoaded } from './jsonl'
import { toError } from './errors'
import type { DatasetName, DatasetRecords } from '../types'

interface RecordsState<T> {
  rows: T[]
  total: number | null
  loading: boolean
  error: Error | null
}

export function useRecords<N extends DatasetName>(name: N, wanted: number): RecordsState<DatasetRecords[N]> {
  const [state, setState] = useState<RecordsState<DatasetRecords[N]>>({
    rows: [], total: null, loading: true, error: null,
  })
  useEffect(() => {
    let live = true
    setState((s) => ({ ...s, loading: true }))
    ensureLoaded(name, wanted).then(
      ({ rows, total }) => live && setState({ rows, total, loading: false, error: null }),
      (error: unknown) => live && setState((s) => ({ ...s, loading: false, error: toError(error) })),
    )
    return () => {
      live = false
    }
  }, [name, wanted])
  return state
}
