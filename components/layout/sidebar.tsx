"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { FacilitiesPopup } from "@/components/facilities/facilities-popup"
import Link from "next/link"
import { Stethoscope } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Plus, MessageSquare, Trash2, Edit2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Input } from "@/components/ui/input"

interface Thread {
  id: string
  title: string
  created_at: string
  updated_at: string
}

interface SidebarProps {
  userId: string
  activeThreadId: string | null
  onThreadSelect: (threadId: string) => void
  onThreadDelete: () => void
  refreshTrigger: number
}

export function Sidebar({ userId, activeThreadId, onThreadSelect, onThreadDelete, refreshTrigger }: SidebarProps) {
  const [threads, setThreads] = useState<Thread[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState("")
  const supabase = createClient()
  const router = useRouter()

  useEffect(() => {
    loadThreads()
  }, [userId, refreshTrigger])

  const loadThreads = async () => {
    try {
      const { data, error } = await supabase
        .from("threads")
        .select("*")
        .eq("user_id", userId)
        .order("updated_at", { ascending: false })

      if (error) throw error
      setThreads(data || [])
    } catch (error) {
      console.error("Error loading threads:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const createNewThread = async () => {
    try {
      const { data, error } = await supabase
        .from("threads")
        .insert({
          user_id: userId,
          title: "New Conversation",
        })
        .select()
        .single()

      if (error) throw error
      setThreads([data, ...threads])
      onThreadSelect(data.id)
    } catch (error) {
      console.error("Error creating thread:", error)
    }
  }

  const deleteThread = async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const { error } = await supabase.from("threads").delete().eq("id", threadId)

      if (error) throw error
      setThreads(threads.filter((t) => t.id !== threadId))
      if (activeThreadId === threadId) {
        onThreadDelete()
      }
    } catch (error) {
      console.error("Error deleting thread:", error)
    }
  }

  const startEditingThread = (thread: Thread, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingThreadId(thread.id)
    setEditingTitle(thread.title)
  }

  const saveThreadTitle = async (threadId: string) => {
    if (!editingTitle.trim()) return

    try {
      const { error } = await supabase.from("threads").update({ title: editingTitle.trim() }).eq("id", threadId)

      if (error) throw error

      setThreads(threads.map((t) => (t.id === threadId ? { ...t, title: editingTitle.trim() } : t)))
      setEditingThreadId(null)
      setEditingTitle("")
    } catch (error) {
      console.error("Error updating thread title:", error)
    }
  }

  const cancelEditing = () => {
    setEditingThreadId(null)
    setEditingTitle("")
  }

  return (
  <div className="bg-slate-900 border-r border-slate-800 flex flex-col flex-1">
      <div className="p-4 border-b border-slate-800">
        <Button onClick={createNewThread} className="w-full bg-blue-600 hover:bg-blue-700">
          <Plus className="mr-2 h-4 w-4" />
          New Chat
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2">
          {isLoading ? (
            <div className="text-slate-400 text-sm p-4">Loading conversations...</div>
          ) : threads.length === 0 ? (
            <div className="text-slate-400 text-sm p-4">No conversations yet</div>
          ) : (
            threads.map((thread) => (
              <div
                key={thread.id}
                className={cn(
                  "group flex items-center justify-between p-3 rounded-lg cursor-pointer hover:bg-slate-800 mb-1",
                  activeThreadId === thread.id && "bg-slate-800",
                )}
                onClick={() => onThreadSelect(thread.id)}
              >
                <div className="flex items-center min-w-0 flex-1">
                  <MessageSquare className="mr-2 h-4 w-4 text-slate-400 flex-shrink-0" />
                  {editingThreadId === thread.id ? (
                    <Input
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          saveThreadTitle(thread.id)
                        } else if (e.key === "Escape") {
                          cancelEditing()
                        }
                      }}
                      onBlur={() => saveThreadTitle(thread.id)}
                      className="h-6 text-xs bg-slate-700 border-slate-600 text-slate-200"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <span className="text-slate-200 text-sm truncate">{thread.title}</span>
                  )}
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 text-slate-400 hover:text-slate-200"
                    onClick={(e) => startEditingThread(thread, e)}
                  >
                    <Edit2 className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 text-slate-400 hover:text-red-400"
                    onClick={(e) => deleteThread(thread.id, e)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
      <div className="p-3 border-t border-slate-800 space-y-4">
        <p className="text-[10px] font-semibold tracking-wide text-slate-500 uppercase px-1">Health Tools</p>
        <FacilitiesPopup />
        <Button
          asChild
          variant="outline"
          className="w-full justify-start gap-2 bg-sidebar-accent/30 border-border text-foreground/90 hover:bg-sidebar-accent/50 hover:text-foreground"
        >
          <Link href="/dashboard/doctor-chat" className="flex items-center gap-2">
            <Stethoscope className="h-4 w-4 text-primary" />
            <span className="font-medium">Doctor Consultations</span>
          </Link>
        </Button>
      </div>
    </div>
  )
}
