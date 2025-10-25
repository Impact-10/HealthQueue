"use client"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { usePathname } from "next/navigation"
import { useEffect, useState } from "react"
import { createClient } from "@/lib/supabase/client"

const nav = [
  { label: "Home", href: "/home" },
  { label: "Myths", href: "/myths" },
  { label: "Dos & Don\'ts", href: "/dos-donts" },
  { label: "First Aid", href: "/first-aid" }
]

export function LandingHeader() {
  const pathname = usePathname()
  const [loading, setLoading] = useState(true)
  const [isAuthed, setIsAuthed] = useState(false)

  useEffect(() => {
    const supabase = createClient()
    let mounted = true
    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return
      setIsAuthed(!!data.session)
      setLoading(false)
    })
    const { data: sub } = supabase.auth.onAuthStateChange((_evt, session) => {
      setIsAuthed(!!session)
    })
    return () => { mounted = false; sub.subscription.unsubscribe() }
  }, [])
  return (
    <header className="w-full border-b border-slate-800/70 bg-gradient-to-b from-slate-950/90 via-slate-900/85 to-slate-900/70 backdrop-blur supports-[backdrop-filter]:bg-slate-900/60 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center gap-8">
        <div className="flex items-center gap-10 flex-1 min-w-0">
          <Link href="/home" className="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-300 via-sky-300 to-emerald-300 bg-clip-text text-transparent whitespace-nowrap">DiagonsAI</Link>
          <nav className="hidden md:flex items-center gap-2 text-[13px] font-medium text-slate-400">
            {nav.map(item => {
              const active = pathname === item.href
              return (
                <Link
                  key={item.label}
                  href={item.href}
                  className={
                    "px-3 py-1.5 rounded-md transition-all duration-200 " +
                    (active
                      ? "bg-slate-800/80 text-slate-100 ring-1 ring-inset ring-slate-700 shadow-sm"
                      : "hover:text-slate-200 hover:bg-slate-800/40")
                  }
                >
                  {item.label}
                </Link>
              )})}
          </nav>
        </div>
        <div className="flex items-center gap-3">
          {loading ? (
            <div className="text-xs text-muted-foreground/70">…</div>
          ) : isAuthed ? (
            <Button asChild className="px-5 bg-indigo-600 hover:bg-indigo-500 shadow">
              <Link href="/dashboard">Dashboard</Link>
            </Button>
          ) : (
            <>
              <Button
                variant="outline"
                asChild
                className="hidden sm:inline-flex border-slate-600/70 bg-slate-800/40 text-slate-300 hover:text-slate-100 hover:bg-slate-700/60 focus-visible:ring-2 focus-visible:ring-indigo-500/60 focus-visible:ring-offset-0 transition-colors"
              >
                <Link href="/auth/login">Sign In</Link>
              </Button>
              <Button asChild className="px-5 bg-indigo-600 hover:bg-indigo-500 shadow">
                <Link href="/auth/signup">Get Started</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
