"use client"

import { useState } from "react"
import { ChatInterface } from "@/components/chat/chat-interface"
import { Sidebar } from "@/components/layout/sidebar"
import { Header } from "@/components/layout/header"
import { UsageDisplay } from "./usage-display"
import { ProfileCard } from "./profile-card"
import type { User as SupabaseUser } from "@supabase/supabase-js"

interface DashboardLayoutProps {
  user: SupabaseUser
  profile: any
}

export function DashboardLayout({ user, profile }: DashboardLayoutProps) {
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const handleThreadSelect = (threadId: string) => {
    setActiveThreadId(threadId)
  }

  const handleNewThread = (threadId: string) => {
    setActiveThreadId(threadId)
    setRefreshTrigger((prev) => prev + 1)
  }

  const handleThreadDelete = () => {
    setActiveThreadId(null)
    setRefreshTrigger((prev) => prev + 1)
  }

  return (
    <div className="flex h-screen bg-slate-950">
      <div className="flex flex-col w-64">
        <Sidebar
          userId={user.id}
          activeThreadId={activeThreadId}
          onThreadSelect={handleThreadSelect}
          onThreadDelete={handleThreadDelete}
          refreshTrigger={refreshTrigger}
        />
        <div className="p-4 border-t border-slate-800 flex flex-col gap-4">
          <UsageDisplay userId={user.id} />
          <ProfileCard user={user} profile={profile} compact={false} />
        </div>
      </div>
      <div className="flex-1 flex flex-col">
        <Header user={user} profile={profile} />
        <main className="flex-1 overflow-hidden">
          <ChatInterface userId={user.id} threadId={activeThreadId} onNewThread={handleNewThread} />
        </main>
      </div>
    </div>
  )
}
