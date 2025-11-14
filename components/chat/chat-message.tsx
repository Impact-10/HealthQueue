"use client"

import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Copy, Download } from "lucide-react"
import { useState } from "react"
import { cn } from "@/lib/utils"
import { MarkdownContent } from "./markdown-content"
import { Badge } from "@/components/ui/badge"
import type { ChatMessageRecord, StructuredResponse, ModelSummary } from "./types"

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

const getStringArray = (value: unknown): string[] | null => {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : null
}

interface ChatMessageProps {
  message: ChatMessageRecord
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const resolveModelLabel = (structured?: StructuredResponse, modelInfo?: ModelSummary) => {
    if (modelInfo) {
      const label = typeof modelInfo.badge === "string" && modelInfo.badge.trim().length > 0
        ? modelInfo.badge.trim()
        : typeof modelInfo.name === "string" && modelInfo.name.trim().length > 0
          ? modelInfo.name.trim()
          : typeof modelInfo.key === "string" && modelInfo.key.trim().length > 0
            ? modelInfo.key.trim()
            : null
      if (label) return label
    }

    if (structured) {
      const metadataName = structured.metadata && typeof structured.metadata.model_name === "string"
        ? structured.metadata.model_name.trim()
        : null
      if (metadataName && metadataName.length > 0) {
        return metadataName
      }

      const modelKey = typeof structured.model === "string" ? structured.model.trim() : null
      if (modelKey && modelKey.length > 0) {
        return modelKey
      }
    }

    return "Model"
  }

  // Render model/mode badges for assistant structured responses
  const renderBadges = (structured?: StructuredResponse, modelInfo?: ModelSummary) => {
    if (!structured && !modelInfo) return null
    const label = resolveModelLabel(structured, modelInfo)
    const mode = modelInfo?.mode || structured?.mode
    const warnings = structured && Array.isArray(structured.warnings) ? structured.warnings : null
    return (
      <div className="flex flex-wrap gap-2 mb-2 items-center">
        <div className="relative group">
          <Badge
            variant="default"
            className="text-base font-bold px-3 py-1 bg-gradient-to-r from-blue-700 to-emerald-600 text-white shadow-lg border-2 border-emerald-400/80 scale-110"
            style={{ letterSpacing: '0.03em' }}
          >
            {label}
            <span className="ml-1 align-middle cursor-pointer" tabIndex={0} title="Model: {label}">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" className="inline-block text-emerald-200 hover:text-white ml-1">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="currentColor" fillOpacity="0.15" />
                <text x="12" y="16" textAnchor="middle" fontSize="13" fill="currentColor">i</text>
              </svg>
            </span>
            <span className="absolute left-1/2 -translate-x-1/2 mt-2 z-10 hidden group-hover:block bg-slate-900 text-xs text-white rounded px-2 py-1 shadow-lg border border-emerald-400/60 min-w-[120px] text-center">
              {typeof modelInfo?.description === 'string' && modelInfo.description.length > 0
                ? modelInfo.description
                : 'AI model used for this answer.'}
            </span>
          </Badge>
        </div>
        {mode && (
          <Badge
            variant={mode === "fallback" ? "destructive" : "outline"}
            className="text-xs font-semibold px-2 py-0.5 border border-emerald-400/60 bg-emerald-900/60 text-emerald-200"
          >
            {mode === "fallback" ? "Fallback" : "Inference"}
          </Badge>
        )}
        {warnings && warnings.length > 0 && (
          <Badge variant="destructive">Warning</Badge>
        )}
      </div>
    )
  }

  const renderContentSections = (content: Record<string, unknown>) => {
    // Handle BERT QA extractive answers
    if (typeof content.answer === "string" && content.answer.trim()) {
      return (
        <div>
          <MarkdownContent content={content.answer} />
          {content.raw_answer && typeof content.raw_answer === "string" && content.raw_answer !== content.answer && (
            <div className="mt-2 text-xs text-slate-400 italic">
              Raw extracted span: "{content.raw_answer}"
            </div>
          )}
          {typeof content.confidence === "number" && (
            <div className="mt-2 text-xs text-slate-400">
              Confidence: {(content.confidence * 100).toFixed(1)}%
            </div>
          )}
          {Array.isArray(content.top_passages) && content.top_passages.length > 0 && (
            <div className="mt-3 pt-3 border-t border-slate-700">
              <div className="text-xs font-semibold text-slate-300 mb-2">Relevant passages:</div>
              {content.top_passages.slice(0, 3).map((passage: any, idx: number) => (
                <div key={idx} className="mb-2 pl-3 border-l-2 border-slate-600">
                  <div className="text-xs text-slate-300">{passage.text}</div>
                  <div className="text-xs text-slate-500 mt-1">
                    Source: {passage.source} | Score: {(passage.score * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )
    }
    
    const summaryValue = content.summary
    if (typeof summaryValue === "string" || summaryValue != null) {
      const summaryText = typeof summaryValue === "string"
        ? summaryValue
        : String(summaryValue)
      return (
        <div>
          <div className="font-semibold mb-1">{summaryText}</div>
          {getStringArray(content.possible_causes)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Possible causes:</span> {getStringArray(content.possible_causes)!.join(", ")}
            </div>
          ) : null}
          {getStringArray(content.recommendations)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Recommendations:</span> {getStringArray(content.recommendations)!.join(", ")}
            </div>
          ) : null}
          {getStringArray(content.follow_up_questions)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Follow-up questions:</span> {getStringArray(content.follow_up_questions)!.join(", ")}
            </div>
          ) : null}
          {getStringArray(content.warning_signs)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Warning signs:</span> {getStringArray(content.warning_signs)!.join(", ")}
            </div>
          ) : null}
        </div>
      )
    }

    if (Array.isArray(content.possible_diagnoses)) {
      const summaryText = typeof summaryValue === "string"
        ? summaryValue
        : String(summaryValue ?? "")
      return (
        <div>
          <div className="font-semibold mb-1">{summaryText}</div>
          {getStringArray(content.possible_diagnoses)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Differential:</span> {getStringArray(content.possible_diagnoses)!.join(", ")}
            </div>
          ) : null}
          {getStringArray(content.rationale)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Rationale:</span> {getStringArray(content.rationale)!.join(", ")}
            </div>
          ) : null}
          {getStringArray(content.recommended_tests)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Recommended tests:</span> {getStringArray(content.recommended_tests)!.join(", ")}
            </div>
          ) : null}
          {getStringArray(content.next_steps)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Next steps:</span> {getStringArray(content.next_steps)!.join(", ")}
            </div>
          ) : null}
          {getStringArray(content.warning_signs)?.length ? (
            <div className="mb-1">
              <span className="font-medium">Warning signs:</span> {getStringArray(content.warning_signs)!.join(", ")}
            </div>
          ) : null}
        </div>
      )
    }

    if (isRecord(content.entities)) {
      const entities = content.entities as Record<string, unknown>
      return (
        <div>
          <div className="font-semibold mb-1">Extracted Entities:</div>
          {entities && typeof entities === "object" && Object.keys(entities).length > 0 ? (
            <ul className="ml-4 list-disc">
              {Object.entries(entities).map(([etype, vals]) => {
                const list = getStringArray(vals)
                return list && list.length > 0 ? (
                  <li key={etype}><span className="font-medium">{etype}:</span> {list.join(", ")}</li>
                ) : null
              }).filter(Boolean)}
            </ul>
          ) : <div className="text-slate-400">No entities found.</div>}
        </div>
      )
    }

    if (Array.isArray(content.record_analysis)) {
      const analysis = content.record_analysis
      return (
        <div>
          <div className="font-semibold mb-1">Clinical Record Analysis:</div>
          {Array.isArray(analysis) && analysis.length > 0 ? (
            <ul className="ml-4 list-disc">
              {analysis.map((item, idx) => {
                if (!isRecord(item)) return null
                const question = typeof item.question === "string" ? item.question : "-"
                const result = isRecord(item.result) ? item.result : undefined
                const contentResult = result && isRecord(result.content) ? result.content : undefined
                const answer = contentResult && typeof contentResult.answer === "string" ? contentResult.answer : "-"
                return (
                  <li key={idx}>
                    <span className="font-medium">Q:</span> {question} <span className="font-medium">A:</span> {answer}
                  </li>
                )
              })}
            </ul>
          ) : <div className="text-slate-400">No analysis available.</div>}
        </div>
      )
    }


    return null
  }

  // Render structured content for assistant
  const renderStructuredContent = (structured?: StructuredResponse) => {
    if (!structured) return <MarkdownContent content={message.content} />
    const { content } = structured
    if (!content || typeof content !== "object") {
      // Fallback: show error if backend returns raw JSON or unexpected format
      return (
        <div className="bg-red-900/80 border border-red-700 text-red-100 rounded p-3 text-sm">
          <span className="font-bold">Error:</span> Unexpected or unstructured response from backend.<br />
          <span className="text-xs break-all">{typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}</span>
        </div>
      )
    }

    if (isRecord(content) && isRecord(content.individual_results)) {
      const individualResults = content.individual_results as Record<string, unknown>
      const entries = Object.entries(individualResults)
      if (entries.length === 0) {
        return <MarkdownContent content={message.content} />
      }
      return (
        <div className="space-y-3">
          <div className="font-semibold">Model comparison</div>
          <div className="grid gap-3 md:grid-cols-2">
            {entries.map(([key, value]) => {
              if (!isRecord(value)) {
                return (
                  <div key={key} className="rounded-lg border border-slate-700 bg-slate-900/40 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold capitalize">{key}</span>
                    </div>
                    <div className="text-sm text-slate-200">No response.</div>
                  </div>
                )
              }

              const valueMode = typeof value.mode === "string" ? value.mode : undefined
              const valueContent = isRecord(value.content) ? value.content : undefined
              const cardContent = valueContent ? renderContentSections(valueContent) : null
              const fallbackSummary = typeof valueContent?.summary === "string"
                ? valueContent.summary
                : typeof valueContent?.answer === "string"
                  ? valueContent.answer
                  : typeof valueContent?.raw_text === "string"
                    ? valueContent.raw_text
                    : ""
              return (
                <div key={key} className="rounded-lg border border-slate-700 bg-slate-900/40 p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold capitalize">{key}</span>
                    {valueMode && (
                      <Badge variant={valueMode === "fallback" ? "destructive" : "outline"}>
                        {valueMode === "fallback" ? "Fallback" : "Inference"}
                      </Badge>
                    )}
                  </div>
                  <div className="text-sm text-slate-200">
                    {cardContent || fallbackSummary || "No response."}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    }

    const sections = renderContentSections(content as Record<string, unknown>)
    if (sections) return sections
    // Fallback: show error if backend returns raw JSON or unexpected format
    return (
      <div className="bg-red-900/80 border border-red-700 text-red-100 rounded p-3 text-sm">
        <span className="font-bold">Error:</span> Unexpected or unstructured response from backend.<br />
        <span className="text-xs break-all">{typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}</span>
      </div>
    )
  }

  return (
    <div className={cn("flex gap-3 group", message.role === "user" ? "justify-end" : "justify-start")}> 
      {message.role === "assistant" && (
        <Avatar className="h-8 w-8 flex-shrink-0">
          <AvatarFallback className="bg-blue-600 text-white text-xs">AI</AvatarFallback>
        </Avatar>
      )}

      <div className={cn("max-w-[80%] space-y-2", message.role === "user" ? "order-first" : "")}> 
        <div
          className={cn(
            "rounded-lg px-4 py-3 text-sm",
            message.role === "user"
              ? "bg-blue-600 text-white ml-auto"
              : "bg-slate-800 text-slate-100 border border-slate-700",
          )}
        >
          {message.role === "assistant" && renderBadges(message.structured, message.modelInfo)}
          {message.role === "assistant"
            ? renderStructuredContent(message.structured)
            : <div className="whitespace-pre-wrap">{message.content}</div>}

          {/* Source Citation for BERT QA responses */}
          {message.role === "assistant" && message.structured && (
            (() => {
              const metadata = message.structured.metadata
              const source = metadata?.source
              if (source && typeof source === "object") {
                const sourceLabel = source.label || source.source_type || "Medical Corpus"
                const sourceQuestion = source.question
                return (
                  <div className="mt-3 pt-3 border-t border-slate-700">
                    <div className="text-xs text-slate-400">
                      <span className="font-semibold text-slate-300">Source:</span> {sourceLabel}
                      {sourceQuestion && (
                        <div className="mt-1 text-slate-500 italic">
                          "{sourceQuestion.length > 100 ? sourceQuestion.substring(0, 100) + "..." : sourceQuestion}"
                        </div>
                      )}
                      {source.retrieval_score && (
                        <div className="mt-1 text-slate-500">
                          Relevance: {(source.retrieval_score * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                )
              }
              return null
            })()
          )}

          {/* Safety Warning */}
          {message.role === "assistant" && message.structured && (
            (() => {
              const metadata = message.structured.metadata
              const warning = metadata?.safety_warning
              if (warning && typeof warning === "string") {
                return (
                  <div className="mt-3 p-2 bg-amber-900/30 border border-amber-700/50 rounded text-xs text-amber-200">
                    ⚠️ {warning}
                  </div>
                )
              }
              return null
            })()
          )}

          {/* Low Confidence Fallback - Top Passages */}
          {message.role === "assistant" && message.structured && (
            (() => {
              const content = message.structured.content
              if (content && typeof content === "object" && "top_passages" in content) {
                const passages = (content as any).top_passages
                if (Array.isArray(passages) && passages.length > 0) {
                  return (
                    <div className="mt-3 space-y-2">
                      <div className="text-xs font-semibold text-slate-300">Top Retrieved Passages:</div>
                      {passages.map((passage: any, idx: number) => (
                        <div key={idx} className="p-2 bg-slate-900/50 rounded border border-slate-700 text-xs">
                          <div className="text-slate-300 mb-1">{passage.text}</div>
                          <div className="text-slate-500 mt-1">
                            Source: {passage.source || "Medical Corpus"} (Rank: {passage.rank})
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                }
              }
              return null
            })()
          )}

          {message.file_urls && message.file_urls.length > 0 && (
            <div className="mt-2 space-y-1">
              {message.file_urls.map((url, index) => (
                <div key={index} className="flex items-center gap-2 text-xs opacity-80">
                  <Download className="h-3 w-3" />
                  <a href={url} target="_blank" rel="noopener noreferrer" className="hover:underline">
                    Attached file {index + 1}
                  </a>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>{formatTime(message.created_at)}</span>
          {message.role === "assistant" && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 hover:bg-slate-700"
              onClick={copyToClipboard}
            >
              <Copy className="h-3 w-3" />
            </Button>
          )}
          {copied && <span className="text-green-400">Copied!</span>}
        </div>
      </div>

      {message.role === "user" && (
        <Avatar className="h-8 w-8 flex-shrink-0">
          <AvatarFallback className="bg-slate-700 text-white text-xs">You</AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}
