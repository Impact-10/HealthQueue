"use client"

import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { LogOut, Settings, User } from "lucide-react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import type { User as SupabaseUser } from "@supabase/supabase-js"

interface HeaderProps {
  user: SupabaseUser
  profile: any
}

export function Header({ user, profile }: HeaderProps) {
  const router = useRouter()
  const supabase = createClient()

  const handleSignOut = async () => {
    await supabase.auth.signOut()
    router.push("/auth/login")
  }

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
  }

  return (
    <header className="border-b border-slate-800 bg-slate-900 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold text-white">DiagonsAI</h1>
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
            <span>Your personal health companion</span>
            <span className="hidden sm:inline-block w-px h-3 bg-slate-700" />
            <Link href="/home" className="hover:text-slate-200 transition-colors">Home</Link>
            <span className="hidden sm:inline-block w-px h-3 bg-slate-700" />
            <Link href="/myths" className="hover:text-slate-200 transition-colors">Myths</Link>
            <span className="hidden sm:inline-block w-px h-3 bg-slate-700" />
            <Link href="/dos-donts" className="hover:text-slate-200 transition-colors">Dos & Don&apos;ts</Link>
            <span className="hidden sm:inline-block w-px h-3 bg-slate-700" />
            <Link href="/first-aid" className="hover:text-slate-200 transition-colors">First Aid</Link>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="relative h-10 w-10 rounded-full">
                <Avatar className="h-10 w-10">
                  <AvatarFallback className="bg-blue-600 text-white">
                    {profile?.full_name ? getInitials(profile.full_name) : getInitials(user.email || "U")}
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56 bg-slate-800 border-slate-700" align="end">
              <DropdownMenuItem className="text-slate-200 focus:bg-slate-700">
                <User className="mr-2 h-4 w-4" />
                <span>Profile</span>
              </DropdownMenuItem>
              <DropdownMenuItem className="text-slate-200 focus:bg-slate-700" onClick={() => router.push('/dashboard/settings')}>
                <Settings className="mr-2 h-4 w-4" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuItem className="text-slate-200 focus:bg-slate-700" onClick={handleSignOut}>
                <LogOut className="mr-2 h-4 w-4" />
                <span>Sign out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  )
}
