import { openAIAdapter } from '@openuidev/react-headless'
import { askAndWait } from '../lib/api.js'
import { nowQuestionDate } from '../lib/format.js'

// Wire format inside the assistant message text. The OpenUI chat only appends
// text, so progress lines and the final result are separated by control chars:
//   \u0001<stage>\n ... \u0002<json result>
export const STAGE_MARK = '\u0001'
export const RESULT_MARK = '\u0002'

export const TOOLS = [
  { name: 'recall', hint: 'Search this person’s conversations and answer with the quotes behind it' },
]

export function messageText(m) {
  if (typeof m?.content === 'string') return m.content
  if (Array.isArray(m?.content)) return m.content.map((p) => p.text || '').join('')
  return ''
}

// "/recall what car does she drive" -> { tool: 'recall', text: 'what car does she drive' }
export function parseSlash(text) {
  const m = /^\s*\/([a-z-]+)\b\s*([\s\S]*)$/i.exec(text || '')
  return m ? { tool: m[1].toLowerCase(), text: m[2].trim() } : null
}

const enc = new TextEncoder()
const chunk = (content) =>
  enc.encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content } }] })}\n\n`)

function sse(run) {
  return new Response(
    new ReadableStream({
      async start(controller) {
        const push = (s) => controller.enqueue(chunk(s))
        try {
          await run(push)
        } catch (e) {
          push(RESULT_MARK + JSON.stringify({ kind: 'error', message: e.message }))
        }
        controller.enqueue(enc.encode('data: [DONE]\n\n'))
        controller.close()
      },
    }),
    { headers: { 'Content-Type': 'text/event-stream' } },
  )
}

// ctx: { split, historyId, getOptions(): {provider}, onResult(result) }
export function makeLLM(ctx) {
  return {
    streamProtocol: openAIAdapter(),
    send: async ({ messages, signal }) => {
      const last = [...messages].reverse().find((m) => m.role === 'user')
      const cmd = parseSlash(messageText(last))
      return sse(async (push) => {
        if (!cmd || cmd.tool !== 'recall') {
          push(RESULT_MARK + JSON.stringify({ kind: 'hint', unknown: cmd?.tool || null }))
          return
        }
        if (!cmd.text) {
          push(RESULT_MARK + JSON.stringify({ kind: 'hint', empty: true }))
          return
        }
        const { provider } = ctx.getOptions()
        const body = {
          split: ctx.split,
          history_id: ctx.historyId,
          question: cmd.text,
          provider,
          question_date: nowQuestionDate(),
        }
        let stage = ''
        const result = await askAndWait(
          body,
          (job) => {
            if (job.stage !== stage) {
              stage = job.stage
              push(`${STAGE_MARK}${stage}\n`)
            }
          },
          signal,
        )
        ctx.onResult(result)
        push(RESULT_MARK + JSON.stringify({ kind: 'recall', ...result }))
      })
    },
  }
}
