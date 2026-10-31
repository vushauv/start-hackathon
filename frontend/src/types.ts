export interface Question {
  question_id: string
  history_id: string
  question: string
  question_date: string
}

export type QuestionType =
  | 'multi-session'
  | 'temporal-reasoning'
  | 'single-session-assistant'
  | 'knowledge-update'
  | 'single-session-user'

export interface Gold {
  question_id: string
  question_type: QuestionType
  answer: string
  abstain: boolean
  aliases: string[]
  evidence: string[]
}

export interface Turn {
  turn_id: string
  role: string
  content: string
}

export interface Session {
  session_id: string
  date: string
  turns: Turn[]
}

export interface History {
  history_id: string
  sessions: Session[]
}

export interface QuestionSelection {
  question: Question
  gold: Gold
}

export interface DatasetRecords {
  questions: Question
  gold: Gold
  histories: History
}

export type DatasetName = keyof DatasetRecords

export interface JsonlIndex {
  size: number
  offsets: number[]
  lengths: number[]
  ids: (string | null)[]
}
