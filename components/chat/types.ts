export interface ModelSummary {
  key: string
  name?: string
  badge?: string
  mode?: string
  [key: string]: unknown
}

export interface StructuredResponse {
  model: string
  mode?: string
  content: Record<string, unknown>
  metadata?: Record<string, unknown>
  warnings?: string[]
  [key: string]: unknown
}

export interface StoredAssistantPayload {
  type: "structured-response"
  version: number
  structuredResponse: StructuredResponse
  text?: string
  modelKey?: string
  directAnswer?: string
  modelInfo?: ModelSummary
  createdAt?: string
  [key: string]: unknown
}

export interface ChatMessageRecord {
  id: string
  role: "user" | "assistant"
  content: string
  file_urls?: string[]
  created_at: string
  structured?: StructuredResponse
  modelInfo?: ModelSummary
}
