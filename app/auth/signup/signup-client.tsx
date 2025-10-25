"use client"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useState, Suspense } from "react"

function SignUpInner() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [fullName, setFullName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault()
    const supabase = createClient()
    setIsLoading(true)
    setError(null)

    if (password !== confirmPassword) {
      setError("Passwords do not match")
      setIsLoading(false)
      return
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters")
      setIsLoading(false)
      return
    }

    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { full_name: fullName } },
      })
      if (error) {
        if (error.message.includes("already registered")) setError("An account with this email already exists")
        else if (error.message.includes("Password")) setError("Password must be at least 6 characters with a mix of letters and numbers")
        else setError(error.message)
        return
      }
      router.replace("/auth/verify-email")
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
            <CardTitle className="text-2xl text-white">Create Account</CardTitle>
            <CardDescription className="text-slate-400">Join DiagonsAI</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSignUp} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="fullName" className="text-slate-200">Full Name</Label>
                <Input id="fullName" type="text" required value={fullName} onChange={(e) => setFullName(e.target.value)} disabled={isLoading} className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-400" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email" className="text-slate-200">Email</Label>
                <Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} disabled={isLoading} className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-400" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-slate-200">Password</Label>
                <Input id="password" type="password" required value={password} onChange={(e) => setPassword(e.target.value)} disabled={isLoading} className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-400" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-slate-200">Confirm Password</Label>
                <Input id="confirmPassword" type="password" required value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} disabled={isLoading} className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-400" />
              </div>
              {error && <div className="text-red-400 text-sm bg-red-950/50 p-3 rounded-md border border-red-900">{error}</div>}
              <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50" disabled={isLoading}>{isLoading ? "Creating account..." : "Create Account"}</Button>
              <div className="mt-6 text-center text-sm text-slate-400">Already have an account? <Link href="/auth/login" className="text-blue-400 hover:text-blue-300 underline underline-offset-4">Sign in</Link></div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export function SignUpClientWrapper() {
  return <Suspense fallback={<div className="p-6 text-center text-slate-300">Loading...</div>}><SignUpInner /></Suspense>
}
