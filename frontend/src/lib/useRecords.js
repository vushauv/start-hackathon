import { useEffect, useState } from 'react'
import { ensureLoaded } from './jsonl.js'

export function useRecords(name, wanted) {
  const [state, setState] = useState({ rows: [], total: null, loading: true, error: null })
  useEffect(() => {
    let live = true
    setState((s) => ({ ...s, loading: true }))
    ensureLoaded(name, wanted).then(
      ({ rows, total }) => live && setState({ rows, total, loading: false, error: null }),
      (error) => live && setState((s) => ({ ...s, loading: false, error })),
    )
    return () => {
      live = false
    }
  }, [name, wanted])
  return state
}
