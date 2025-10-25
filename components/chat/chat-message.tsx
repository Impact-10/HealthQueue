"use client"

import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Copy, Download } from "lucide-react"
import { useState } from "react"
import { cn } from "@/lib/utils"
import { MarkdownContent } from "./markdown-content"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  file_urls?: string[]
  created_at: string
}

interface ChatMessageProps {
  message: Message
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
          {message.role === "assistant" ? (
            <MarkdownContent content={message.content} />
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
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
