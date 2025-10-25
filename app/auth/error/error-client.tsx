"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { useSearchParams, } from "next/navigation"
import { Suspense } from "react"

function AuthErrorInner() {
  const searchParams = useSearchParams()
  const error = searchParams.get("error")
  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl text-white">Authentication Error</CardTitle>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            {error ? <p className="text-red-400 bg-red-950/50 p-3 rounded-md border border-red-900">{error}</p> : <p className="text-slate-300">An authentication error occurred.</p>}
            <Button asChild className="w-full bg-blue-600 hover:bg-blue-700"><Link href="/auth/login">Back to Sign In</Link></Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export function AuthErrorClient() {
  return <Suspense fallback={<div className="p-6 text-center text-slate-300">Loading...</div>}><AuthErrorInner /></Suspense>
}
