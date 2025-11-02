/* eslint-disable @typescript-eslint/no-explicit-any */
"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback, useMemo } from "react"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Send, Paperclip, Loader2, MessageSquare, AlertTriangle, Mic, MicOff } from "lucide-react"
import { useSpeechInput } from "@/hooks/use-speech-input"
import { useTextToSpeech } from "@/hooks/use-text-to-speech"
import { ChatReportDialog } from "./chat-report-dialog"
import { ChatMessage } from "./chat-message"
import { useVirtualizer } from "@tanstack/react-virtual"
import { FileUpload } from "./file-upload"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ClientOnly } from "../client-only";

import type {
  ChatMessageRecord,
  StructuredResponse,
  StoredAssistantPayload,
  ModelSummary,
} from "./types"

interface ChatInterfaceProps {
  userId: string
  threadId?: string | null
  onNewThread?: (threadId: string) => void
}


const MODEL_OPTIONS = [
  {
    value: "distilgpt2",
    label: "MedAlpaca (DistilGPT-2)",
    description: "Fast, lightweight open-source model for medical Q&A.",
  },
  {
    value: "gemini",
    label: "Gemini (Google API)",
    description: "Google Gemini Pro for medical Q&A (API key required).",
  },
  {
    value: "biogpt",
    label: "BioGPT (Diabetology)",
    description: "Generative biomedical model tuned for diabetes-focused Q&A.",
  },
] as const;

type ModelKey = "distilgpt2" | "gemini" | "biogpt"; // add biogpt
const DEFAULT_MODEL: ModelKey = "distilgpt2";

const MODEL_LOOKUP: Record<ModelKey, (typeof MODEL_OPTIONS)[number]> = {
  distilgpt2: MODEL_OPTIONS[0],
  gemini: MODEL_OPTIONS[1],
  biogpt: MODEL_OPTIONS[2],
};


const STRUCTURED_PAYLOAD_TYPE = "structured-response" as const

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string")

const isEntityMap = (value: unknown): value is Record<string, string[]> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  return Object.values(value).every((entry) => isStringArray(entry))
}

const deriveDisplayText = (structured?: StructuredResponse | null): string => {
  if (!structured) return ""
  const content = structured.content ?? {}
  if (!content || typeof content !== "object") return structured.model

  const summary = typeof (content as Record<string, unknown>).summary === "string"
    ? (content as Record<string, unknown>).summary?.toString().trim()
    : ""
  if (summary) return summary

  const answer = typeof (content as Record<string, unknown>).answer === "string"
    ? (content as Record<string, unknown>).answer?.toString().trim()
    : ""
  if (answer) return answer

  const possibleCauses = (content as Record<string, unknown>).possible_causes
  if (isStringArray(possibleCauses) && possibleCauses.length > 0) {
    const preview = possibleCauses.slice(0, 3).join(", ")
    return `Possible causes: ${preview}${possibleCauses.length > 3 ? "…" : ""}`
  }

  const entitiesValue = (content as Record<string, unknown>).entities
  if (isEntityMap(entitiesValue)) {
    const summaryItems = Object.entries(entitiesValue)
      .filter(([, items]) => items.length > 0)
      .slice(0, 3)
      .map(([key, items]) => `${key}: ${items.slice(0, 2).join(", ")}${items.length > 2 ? "…" : ""}`)
    if (summaryItems.length > 0) {
      return `Entities detected — ${summaryItems.join("; ")}`
    }
  }

  return structured.model
}

const parseAssistantPayload = (
  raw: string,
): { structured?: StructuredResponse; text: string; modelInfo?: ModelSummary } => {
  if (!raw) return { text: "" }

  try {
    const parsed = JSON.parse(raw) as StoredAssistantPayload | StructuredResponse

    if (parsed && typeof parsed === "object" && "type" in parsed) {
      const payload = parsed as StoredAssistantPayload
      if (payload.type === STRUCTURED_PAYLOAD_TYPE && payload.structuredResponse) {
        const textCandidates: Array<string | undefined> = [
          payload.text,
          payload.directAnswer,
          deriveDisplayText(payload.structuredResponse),
        ]
        const text = textCandidates.find((candidate) => typeof candidate === "string" && candidate.trim().length > 0)
          ?.trim() ?? ""
        return {
          structured: payload.structuredResponse,
          text,
          modelInfo: payload.modelInfo,
        }
      }
    }

    if (parsed && typeof parsed === "object" && "model" in parsed && "content" in parsed) {
      const structured = parsed as StructuredResponse
      return {
        structured,
        text: deriveDisplayText(structured),
      }
    }
  } catch (error) {
    console.error("Failed to parse assistant payload", error)
  }

  return { text: raw }
}

export function ChatInterface({ userId, threadId, onNewThread }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessageRecord[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [showFileUpload, setShowFileUpload] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [rateLimitError, setRateLimitError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelKey>(DEFAULT_MODEL)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const supabase = createClient()
  const activeModel = MODEL_LOOKUP[selectedModel]
  const handleModelChange = (value: string) => setSelectedModel(value as ModelKey)
  // Scroll position persistence per thread
  const scrollPositionsRef = useRef<Record<string, number>>({})
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)
  const [showNewMessages, setShowNewMessages] = useState(false)
  const { supported: speechSupported, listening, interim, final, start: startSpeech, stop: stopSpeech, reset: resetSpeech } = useSpeechInput({ interim: true })
  const { supported: ttsSupported, speak } = useTextToSpeech()
  // Flag indicating next assistant response should be spoken
  const speakNextRef = useRef(false)

  // Merge final + interim into visible input when listening; keep user manual edits safe if they type mid-stream.
  useEffect(() => {
    if (!listening) return
    setInput(() => {
      const assembled = [final, interim].filter(Boolean).join(' ').replace(/\s+/g,' ').trim()
      return assembled
    })
  }, [final, interim, listening])

  // On stop listening, ensure only final committed (interim discarded)
  useEffect(() => {
    if (!listening && final) {
      setInput(prev => prev) // no-op; placeholder for future normalization
    }
  }, [listening, final])

  const toggleMic = () => {
    if (!speechSupported) return
    if (listening) {
      stopSpeech()
    } else {
      // Clear existing staged transcript if starting fresh
      resetSpeech()
      startSpeech()
      // Mark that when this user message is sent, we should TTS the assistant reply
      speakNextRef.current = true
    }
  }

  // Virtualization threshold (below this, normal render avoids overhead)
  const VIRTUALIZE_AFTER = 60
  const enableVirtual = messages.length > VIRTUALIZE_AFTER

  // Virtualizer setup
  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => scrollContainerRef.current,
    estimateSize: () => 120, // rough average height; dynamic content will adjust as measured
    overscan: 12,
  })

  // Load messages & restore scroll position
  const loadMessages = useCallback(async () => {
    if (!threadId) {
      setMessages([])
      return
    }

    try {
      const { data, error } = await supabase
        .from("messages")
        .select("*")
        .eq("thread_id", threadId)
        .order("created_at", { ascending: true })

      if (error) throw error
      const mapped = (data || []).map((message) => {
        const base = message as ChatMessageRecord
        if (base.role === "assistant") {
          const parsed = parseAssistantPayload(base.content)
          return {
            ...base,
            content: parsed.text,
            structured: parsed.structured,
            modelInfo: parsed.modelInfo ?? base.modelInfo,
          }
        }
        return base
      })
      setMessages(mapped)
    } catch (error) {
      console.error("Error loading messages:", error)
    }
  }, [supabase, threadId])

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      if (threadId) {
        await loadMessages()
        // Restore scroll top after a frame so content height is laid out
        requestAnimationFrame(() => {
          if (cancelled) return
          const el = scrollContainerRef.current
          if (el && threadId && scrollPositionsRef.current[threadId] != null) {
            el.scrollTop = scrollPositionsRef.current[threadId]
            const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
            autoScrollRef.current = atBottom
          } else {
            autoScrollRef.current = true
            messagesEndRef.current?.scrollIntoView({ behavior: "auto" })
          }
        })
      } else {
        setMessages([])
      }
    }
    run()
    return () => {
      cancelled = true
    }
  }, [threadId, loadMessages])

  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    const handler = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      autoScrollRef.current = atBottom
      if (atBottom) {
        setShowNewMessages(false)
      }
      if (threadId) {
        scrollPositionsRef.current[threadId] = el.scrollTop
      }
    }
    el.addEventListener("scroll", handler, { passive: true })
    return () => el.removeEventListener("scroll", handler)
  }, [threadId])

  useEffect(() => {
    if (autoScrollRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    } else {
      // User is reading earlier content; show indicator
      setShowNewMessages(true)
    }
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() && uploadedFiles.length === 0) return

    setIsLoading(true)
    setRateLimitError(null)
    const userMessage = input.trim()
    setInput("")
    setUploadedFiles([])

    try {
      const tempUserMessage: ChatMessageRecord = {
        id: `temp-${Date.now()}`,
        role: "user",
        content: userMessage,
        file_urls: uploadedFiles.length > 0 ? uploadedFiles : undefined,
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, tempUserMessage])

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          fileUrls: uploadedFiles,
          threadId,
          userId,
          modelKey: selectedModel,
        }),
      })

      if (response.status === 429) {
        const errorData = await response.json()
        setRateLimitError(errorData.error)
        setMessages((prev) => prev.slice(0, -1))
        return
      }

      if (!response.ok) {
  let detail = "Failed to send message";
  try {
    const err = await response.json();
    detail = err?.error || err?.body || JSON.stringify(err);
  } catch {
    try { detail = await response.text(); } catch {}
  }
  console.error("Chat API 502 detail:", detail);
  setRateLimitError(typeof detail === "string" ? detail : "Service unavailable");
  throw new Error(detail);
}
      const data = await response.json()

      const directAnswer = typeof data.directAnswer === "string" && data.directAnswer.trim().length > 0
        ? data.directAnswer.trim()
        : typeof data.answer === "string" && data.answer.trim().length > 0
          ? data.answer.trim()
          : undefined
      const structured: StructuredResponse | undefined = data.structuredResponse
      const displayText = directAnswer
        || (structured ? deriveDisplayText(structured) : "")
        || (typeof data.response === "string" ? data.response : "")

      const backendModelInfo = (data.modelInfo as ModelSummary | undefined) ?? undefined
      const derivedModelInfo = backendModelInfo
        ?? (() => {
          if (structured && typeof structured.model === "string") {
            const normalized = structured.model.trim()
            if (normalized.length > 0) {
              return {
                key: normalized,
                name: normalized,
                badge: normalized,
                mode: structured.mode,
              } as ModelSummary
            }
          }
          const option = MODEL_LOOKUP[selectedModel]
          if (option) {
            return {
              key: selectedModel,
              name: option.label,
              badge: option.label,
            } as ModelSummary
          }
          return undefined
        })()

      const assistantMessage: ChatMessageRecord = {
        id: data.messageId || `temp-assistant-${Date.now()}`,
        role: "assistant",
        content: displayText,
        structured,
        modelInfo: derivedModelInfo,
        created_at: new Date().toISOString(),
      }

      setMessages((prev) => [...prev.slice(0, -1), tempUserMessage, assistantMessage])

      // If user used mic for this turn and TTS available, speak assistant reply
      if (ttsSupported && speakNextRef.current) {
        speakNextRef.current = false
        // Basic sanitization: collapse whitespace
        const toSpeak = assistantMessage.content.replace(/\s+/g, ' ').trim()
        if (toSpeak) speak(toSpeak)
      } else {
        // Reset if not spoken (e.g., user typed instead)
        speakNextRef.current = false
      }

      if (data.threadId && data.threadId !== threadId && onNewThread) {
        onNewThread(data.threadId)
      }
    } catch (error) {
      console.error("Error sending message:", error)
      setMessages((prev) => prev.slice(0, -1))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleFileUpload = (urls: string[]) => {
    setUploadedFiles((prev) => [...prev, ...urls])
    setShowFileUpload(false)
  }

  const scrollToBottom = useCallback(() => {
    const el = scrollContainerRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
    autoScrollRef.current = true
    setShowNewMessages(false)
  }, [])

  const renderedMessages = useMemo(() => {
    if (!enableVirtual) {
      return messages.map((message) => <ChatMessage key={message.id} message={message} />)
    }
    const virtualItems = virtualizer.getVirtualItems()
    return (
      <div
        style={{
          height: virtualizer.getTotalSize(),
          position: "relative",
        }}
      >
        {virtualItems.map((vi: { index: number; start: number }) => {
          const msg = messages[vi.index]
          return (
            <div
              key={msg.id}
              data-index={vi.index}
              ref={virtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${vi.start}px)`,
              }}
            >
              <ChatMessage message={msg} />
            </div>
          )
        })}
      </div>
    )
  }, [messages, enableVirtual, virtualizer])

  return (
    <div className="flex flex-col h-full">
      <div className="max-w-4xl mx-auto w-full pt-2 pb-1 px-2">
        <div className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4">
          <div className="flex-1 min-w-0">
            <ClientOnly>
              <Select value={selectedModel} onValueChange={handleModelChange}>
                <SelectTrigger className="w-full md:w-72 bg-slate-800 border-slate-700 text-white">
                  <SelectValue placeholder="Select AI Model" />
                </SelectTrigger>
                <SelectContent>
                  {MODEL_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="flex flex-col gap-0.5">
                      <span className="font-medium">{option.label}</span>
                      <span className="text-xs text-slate-400">{option.description}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </ClientOnly>
          </div>
          <div className="flex-1 min-w-0 hidden md:block">
            <span className="text-xs text-slate-400">{activeModel?.description}</span>
          </div>
        </div>
      </div>
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-4 pr-2 scrollbar-thin scrollbar-track-slate-900 scrollbar-thumb-slate-700 hover:scrollbar-thumb-slate-600"
      >
        <div className="max-w-4xl mx-auto space-y-4">
          {threadId && messages.length > 0 && (
            <div className="flex justify-end">
              <ChatReportDialog threadId={threadId} userId={userId} />
            </div>
          )}
          {rateLimitError && (
            <Alert className="bg-amber-950/50 border-amber-900">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-amber-200">{rateLimitError}</AlertDescription>
            </Alert>
          )}

          {messages.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-slate-400 mb-4">
                <MessageSquare className="mx-auto h-12 w-12 mb-4" />
                <h3 className="text-lg font-medium text-slate-300">
                  {threadId ? "Start the conversation" : "Select a conversation or start a new one"}
                </h3>
                <p className="text-sm">
                  {threadId
                    ? "Ask me anything about your health and wellness"
                    : "Choose from the sidebar or create a new chat"}
                </p>
              </div>
            </div>
          ) : (
            renderedMessages
          )}
          {isLoading && (
            <div className="flex justify-center">
              <div className="flex items-center gap-2 text-slate-400">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>AI is thinking...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {showNewMessages && (
        <div className="absolute bottom-28 left-1/2 -translate-x-1/2 z-10">
          <Button
            size="sm"
            variant="secondary"
            className="shadow-md bg-slate-800/80 backdrop-blur border border-slate-700 hover:bg-slate-700"
            onClick={scrollToBottom}
          >
            New messages ↓
          </Button>
        </div>
      )}

      <div className="border-t border-slate-800 p-4">
        <div className="max-w-4xl mx-auto">
          {uploadedFiles.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {uploadedFiles.map((url, index) => (
                <div key={index} className="bg-slate-800 px-2 py-1 rounded text-xs text-slate-300">
                  File attached
                  <button
                    onClick={() => setUploadedFiles((prev) => prev.filter((_, i) => i !== index))}
                    className="ml-2 text-slate-400 hover:text-slate-200"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          <form onSubmit={handleSubmit} className="flex gap-2">
            <div className="flex-1 relative">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your health..."
                className="min-h-[60px] max-h-32 resize-none bg-slate-800 border-slate-700 text-white placeholder:text-slate-400 pr-24"
                disabled={isLoading}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-2 top-2 h-8 w-8 p-0 text-slate-400 hover:text-slate-200"
                onClick={() => setShowFileUpload(true)}
              >
                <Paperclip className="h-4 w-4" />
              </Button>
              {speechSupported && (
                <Button
                  type="button"
                  variant={listening ? "default" : "ghost"}
                  size="sm"
                  aria-pressed={listening}
                  className={
                    "absolute top-2 h-8 w-8 p-0 text-slate-400 hover:text-slate-200 transition-colors " +
                    (listening
                      ? "right-12 bg-rose-600 hover:bg-rose-500 text-white animate-pulse"
                      : "right-12")
                  }
                  onClick={toggleMic}
                  title={listening ? "Stop voice input" : "Start voice input"}
                >
                  {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </Button>
              )}
            </div>
            <Button
              type="submit"
              disabled={isLoading || (!input.trim() && uploadedFiles.length === 0)}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </div>
      </div>

      {showFileUpload && <FileUpload onUpload={handleFileUpload} onClose={() => setShowFileUpload(false)} />}
    </div>
  )
}
