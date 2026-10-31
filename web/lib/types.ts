// Shapes of `pipeline serve` (pipeline/pipeline/server.py) responses.

export type Provider = "reactor" | "groq" | "openai";

export interface Settings {
  reranker: string; // base | v2-m3 | none
  rerank_text: "best1" | "best2";
  fusion: "fused" | "rerank";
  rerank_threshold: number | null; // null = per-type top-k
  provider: Provider;
  model: string;
  max_chars: number | null;
  attempts: number | null;
  timeout: number | null;
  dry_run: boolean;
  show_message: boolean;
}

export interface HistorySummary {
  id: string;
  questions: number;
  first_question: string;
}

export interface Meta {
  split: string;
  device: string;
  has_gold: boolean;
  embedder: string;
  rerankers: string[];
  histories: HistorySummary[];
  defaults: Settings;
  provider_defaults: Record<Provider, { model: string; max_chars: number }>;
}

export interface DatasetQuestion {
  id: string;
  question: string;
  date: string;
  type: string;
}

export interface HistoryDetail {
  id: string;
  sessions: number;
  b_chunks: number;
  f_chunks: number;
  question_date: string;
  index_ms: number;
  questions: DatasetQuestion[];
}

export interface SessionRef {
  sid: string;
  date: string;
  gold: boolean;
}

export interface EmbeddingData {
  model: string;
  prefix: string;
  dim: number;
  norm: number;
  head: number[];
}

export interface FirstStageData {
  rrf_k: number;
  channels: { name: string; label: string; top: (SessionRef & { score: number })[] }[];
  candidates: (SessionRef & { rrf: number; ranks: Record<string, number | null> })[];
}

export interface RerankData {
  reranker: string;
  fusion: string;
  rerank_text: string;
  order: (SessionRef & {
    score: number | null;
    rerank_rank: number | null;
    rrf_rank: number;
    moved: number;
    selected: boolean;
  })[];
}

export interface SelectionData {
  guessed_type: string;
  gold_type: string | null;
  k: number;
  rule: string;
  question_date: string;
  hint: string;
}

export interface Passage {
  label: string;
  id: string;
  date: string;
  chars: number;
  text: string;
  gold: boolean;
}

export interface EvidenceData {
  chars: number;
  cap: number;
  passages: Passage[];
}

export interface ComposeData {
  chars: number;
  limit: number;
  passages: number;
  notes_chars: number;
  retrieval_ms: number;
  message: string | null;
}

export interface StepMap {
  embedding: EmbeddingData;
  first_stage: FirstStageData;
  rerank: RerankData;
  selection: SelectionData;
  evidence: EvidenceData;
  compose: ComposeData;
}

export type StepId = keyof StepMap;

export const STEP_ORDER: StepId[] = ["embedding", "first_stage", "rerank", "selection", "evidence", "compose"];

export interface Citation {
  label: string;
  id: string;
  date: string;
}

export interface AnswerEvent {
  type: "answer";
  status: "answered" | "abstained" | "no-answer" | "dry-run";
  who: string;
  answer: string;
  reply?: string;
  reasoning?: string;
  latency_s?: number;
  attempts?: number;
  error?: string;
  citations: Citation[];
}

export interface GoldEvent {
  type: "gold";
  abstain: boolean;
  answer: string;
  evidence: string[];
  question_type: string | null;
}

export type StreamEvent =
  | { type: "queued" }
  | { type: "start"; history_id: string; question: string; question_date: string; question_id: string | null; settings: Settings }
  | { [K in StepId]: { type: "step"; id: K; ms: number; data: StepMap[K] } }[StepId]
  | { type: "waiting"; who: string; expect: string }
  | AnswerEvent
  | GoldEvent
  | { type: "done"; total_ms: number }
  | { type: "error"; message: string };

export interface Turn {
  id: string;
  question: string;
  questionId: string | null;
  historyId: string;
  phase: "queued" | "retrieving" | "waiting" | "done" | "error";
  questionDate?: string;
  steps: Partial<{ [K in StepId]: { ms: number; data: StepMap[K] } }>;
  waiting?: { who: string; expect: string };
  answer?: AnswerEvent;
  gold?: GoldEvent;
  error?: string;
  totalMs?: number;
  startedAt: number;
}
