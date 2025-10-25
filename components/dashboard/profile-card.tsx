"use client"

import { createClient } from "@/lib/supabase/client"
import type { User as SupabaseUser } from "@supabase/supabase-js"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { LogOut, Mail } from "lucide-react"
import { useState } from "react"
import { useRouter } from "next/navigation"

interface ProfileCardProps {
  user: SupabaseUser
  profile: any
  compact?: boolean
}

export function ProfileCard({ user, profile, compact = false }: ProfileCardProps) {
  const supabase = createClient()
  const router = useRouter()
  const [loading, setLoading] = useState(false)

  const fullName = profile?.full_name || user.user_metadata?.full_name || null
  const email = user.email

const initials: string = (fullName || email || "U")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p: string): string => p[0]!.toUpperCase())
    .join("")

  const handleSignOut = async () => {
    try {
      setLoading(true)
      await supabase.auth.signOut()
      router.replace("/auth/login")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <Avatar className="h-12 w-12">
          <AvatarFallback className="bg-blue-600 text-white text-sm">{initials}</AvatarFallback>
        </Avatar>
        <div className="min-w-0">
          {fullName && <p className="text-slate-200 font-medium truncate">{fullName}</p>}
          {email && (
            <p className="text-slate-400 text-xs flex items-center gap-1 truncate">
              <Mail className="h-3 w-3" /> {email}
            </p>
          )}
          {!fullName && !email && <p className="text-slate-500 text-sm">Unknown user</p>}
        </div>
      </div>
      {!compact && (
        <div className="flex gap-2">
          <Button
            onClick={handleSignOut}
            disabled={loading}
            size="sm"
            variant="outline"
            className="border-slate-700 text-slate-300 hover:bg-slate-800 flex-1"
          >
            <LogOut className="h-4 w-4 mr-1" /> {loading ? "Signing out..." : "Sign out"}
          </Button>
        </div>
      )}
    </div>
  )
}
