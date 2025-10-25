"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export function VerifyEmailClient() {
  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl text-white">Check Your Email</CardTitle>
            <CardDescription className="text-slate-400">We've sent you a verification link</CardDescription>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            <div className="space-y-3">
              <p className="text-slate-300">Please check your email and click the verification link to activate your account.</p>
              <p className="text-sm text-slate-400">The link will expire in 24 hours for security.</p>
              <div className="bg-slate-800/50 p-3 rounded-md border border-slate-700">
                <p className="text-xs text-slate-400"><strong>Didn't receive the email?</strong><br />Check your spam folder or contact support if you continue having issues.</p>
              </div>
            </div>
            <div className="space-y-2">
              <Button asChild className="w-full bg-blue-600 hover:bg-blue-700"><Link href="/auth/login">Back to Sign In</Link></Button>
              <Button asChild variant="outline" className="w-full border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent"><Link href="/auth/signup">Try Again</Link></Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
