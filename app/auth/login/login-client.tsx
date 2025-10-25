"use client"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { useState, Suspense } from "react"

function LoginInner() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()
  const searchParams = useSearchParams()
  const nextPath = searchParams.get("next") || "/dashboard"

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    const supabase = createClient()
    setIsLoading(true)
    setError(null)
    try {
      const { error } = await supabase.auth.signInWithPassword({ email, password })
      if (error) {
        if (error.message.includes("Invalid login credentials")) setError("Invalid email or password")
        else if (error.message.includes("Email not confirmed")) setError("Please check your email and confirm your account")
        else setError(error.message)
        return
      }
      router.replace(nextPath)
    } catch {
      setError("An unexpected error occurred. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl text-white">Welcome Back</CardTitle>
            <CardDescription className="text-slate-400">Sign in to DiagonsAI</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-slate-200">Email</Label>
                <Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} disabled={isLoading} className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-400" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-slate-200">Password</Label>
                <Input id="password" type="password" required value={password} onChange={(e) => setPassword(e.target.value)} disabled={isLoading} className="bg-slate-800 border-slate-700 text-white" />
              </div>
              {error && <div className="text-red-400 text-sm bg-red-950/50 p-3 rounded-md border border-red-900">{error}</div>}
              <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50" disabled={isLoading}>{isLoading ? "Signing in..." : "Sign In"}</Button>
              <div className="mt-4 text-center text-sm text-slate-400">Don't have an account? <Link href="/auth/signup" className="text-blue-400 hover:text-blue-300 underline underline-offset-4">Sign up</Link></div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export function LoginClientWrapper() {
  return <Suspense fallback={<div className="p-6 text-center text-slate-300">Loading...</div>}><LoginInner /></Suspense>
}
