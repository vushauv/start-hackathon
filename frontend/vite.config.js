import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const here = path.dirname(fileURLToPath(import.meta.url))
const DATA_DIR = path.resolve(process.env.DATA_DIR || path.join(here, '..'))
const NAMES = new Set(['histories', 'questions', 'gold'])

// name -> Promise<{ size, offsets, lengths, ids }>
const indexCache = new Map()

// Scans a JSONL file once and records the byte offset/length of every line,
// plus its leading id, so the browser can fetch single records with Range requests.
function buildIndex(file) {
  return new Promise((resolve, reject) => {
    const offsets = []
    const lengths = []
    let pos = 0
    let lineStart = 0
    fs.createReadStream(file)
      .on('data', (buf) => {
        let i = buf.indexOf(10)
        let from = 0
        while (i !== -1) {
          const end = pos + i
          if (end > lineStart) {
            offsets.push(lineStart)
            lengths.push(end - lineStart)
          }
          lineStart = end + 1
          from = i + 1
          i = buf.indexOf(10, from)
        }
        pos += buf.length
      })
      .on('end', () => {
        if (pos > lineStart) {
          offsets.push(lineStart)
          lengths.push(pos - lineStart)
        }
        const fd = fs.openSync(file, 'r')
        const head = Buffer.alloc(96)
        const ids = offsets.map((o) => {
          const n = fs.readSync(fd, head, 0, 96, o)
          const m = /^\{"[a-z_]+":\s*"([^"]+)"/.exec(head.toString('utf8', 0, n))
          return m ? m[1] : null
        })
        fs.closeSync(fd)
        resolve({ size: pos, offsets, lengths, ids })
      })
      .on('error', reject)
  })
}

function serveData(req, res, next) {
  const m = /^\/data\/([a-z]+)\.jsonl(\.index)?(?:\?.*)?$/.exec(req.url || '')
  if (!m || !NAMES.has(m[1])) return next()
  const file = path.join(DATA_DIR, `${m[1]}.jsonl`)
  if (!fs.existsSync(file)) {
    res.statusCode = 404
    return res.end(`${m[1]}.jsonl not found in ${DATA_DIR}`)
  }
  if (m[2]) {
    if (!indexCache.has(m[1])) indexCache.set(m[1], buildIndex(file))
    return indexCache.get(m[1]).then(
      (idx) => {
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(idx))
      },
      (err) => {
        indexCache.delete(m[1])
        res.statusCode = 500
        res.end(String(err))
      },
    )
  }
  const size = fs.statSync(file).size
  const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '')
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8')
  res.setHeader('Accept-Ranges', 'bytes')
  let opts
  if (range) {
    const start = Number(range[1])
    const end = range[2] ? Math.min(Number(range[2]), size - 1) : size - 1
    res.statusCode = 206
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`)
    res.setHeader('Content-Length', end - start + 1)
    opts = { start, end }
  } else {
    res.setHeader('Content-Length', size)
  }
  fs.createReadStream(file, opts).pipe(res)
}

const dataPlugin = {
  name: 'jsonl-data',
  configureServer(s) {
    s.middlewares.use(serveData)
  },
  configurePreviewServer(s) {
    s.middlewares.use(serveData)
  },
}

export default defineConfig({
  plugins: [react(), dataPlugin],
  server: {
    proxy: { '/api': { target: process.env.API_URL || 'http://127.0.0.1:8001', changeOrigin: true } },
    watch: process.env.USE_POLLING ? { usePolling: true, interval: 300 } : undefined,
  },
})
