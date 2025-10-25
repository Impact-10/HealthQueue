"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback, useMemo } from "react"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, Paperclip, Loader2, MessageSquare, AlertTriangle, Mic, MicOff } from "lucide-react"
import { useSpeechInput } from "@/hooks/use-speech-input"
import { useTextToSpeech } from "@/hooks/use-text-to-speech"
import { ChatReportDialog } from "./chat-report-dialog"
import { ChatMessage } from "./chat-message"
import { useVirtualizer } from "@tanstack/react-virtual"
import { FileUpload } from "./file-upload"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  file_urls?: string[]
  created_at: string
}

interface ChatInterfaceProps {
  userId: string
  threadId?: string | null
  onNewThread?: (threadId: string) => void
}

export function ChatInterface({ userId, threadId, onNewThread }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [showFileUpload, setShowFileUpload] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [rateLimitError, setRateLimitError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const supabase = createClient()
  // Scroll position persistence per thread
  const scrollPositionsRef = useRef<Record<string, number>>({})
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)
  const [showNewMessages, setShowNewMessages] = useState(false)
  const { supported: speechSupported, listening, interim, final, start: startSpeech, stop: stopSpeech, reset: resetSpeech } = useSpeechInput({ interim: true })
  const { supported: ttsSupported, speak, cancel: cancelTTS } = useTextToSpeech()
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
  }, [threadId])

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
  }, [])

  useEffect(() => {
    if (autoScrollRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    } else {
      // User is reading earlier content; show indicator
      setShowNewMessages(true)
    }
  }, [messages])

  const loadMessages = async () => {
    if (!threadId) return

    try {
      const { data, error } = await supabase
        .from("messages")
        .select("*")
        .eq("thread_id", threadId)
        .order("created_at", { ascending: true })

      if (error) throw error
      setMessages(data || [])
    } catch (error) {
      console.error("Error loading messages:", error)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() && uploadedFiles.length === 0) return

    setIsLoading(true)
    setRateLimitError(null)
    const userMessage = input.trim()
    setInput("")
    setUploadedFiles([])

    try {
      const tempUserMessage: Message = {
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
        }),
      })

      if (response.status === 429) {
        const errorData = await response.json()
        setRateLimitError(errorData.error)
        setMessages((prev) => prev.slice(0, -1))
        return
      }

      if (!response.ok) {
        throw new Error("Failed to send message")
      }

      const data = await response.json()

      const assistantMessage: Message = {
        id: data.messageId || `temp-assistant-${Date.now()}`,
        role: "assistant",
        content: data.response,
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
        {virtualItems.map((vi) => {
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
