async function j(url, opts) {
  const r = await fetch(url, opts)
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const b = await r.json()
      if (b.detail) msg = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail)
    } catch {
      /* not json */
    }
    throw new Error(msg)
  }
  return r.json()
}

const qs = (o) => new URLSearchParams(o).toString()

export const getMeta = () => j('/api/meta')
export const getPeople = (split) => j(`/api/people?${qs({ split })}`)
export const getSession = (split, history_id, session_id) =>
  j(`/api/session?${qs({ split, history_id, session_id })}`)

// Starts a job and polls it; onStage({stage, elapsed_s}) is called on every poll.
export async function askAndWait(body, onStage, signal) {
  const { job_id } = await j('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  for (;;) {
    if (signal?.aborted) throw new Error('cancelled')
    const job = await j(`/api/jobs/${job_id}`)
    onStage?.(job)
    if (job.stage === 'done') return job.result
    if (job.stage === 'error') throw new Error(job.error || 'the question failed')
    await new Promise((r) => setTimeout(r, 1200))
  }
}
export const getHistory = (split, history_id) => j(`/api/history?${qs({ split, history_id })}`)
