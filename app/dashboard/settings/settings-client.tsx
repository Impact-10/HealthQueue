"use client"

import { useState } from "react"
import type { User } from "@supabase/supabase-js"
import { Button } from "@/components/ui/button"
import { HealthProfileForm } from "@/components/dashboard/health-profile-form"
import { Input } from "@/components/ui/input"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface HealthProfile {
  age?: number | null
  gender?: string | null
  medications?: string | null
  conditions?: string | null
  allergies?: string | null
  height_cm?: number | null
  weight_kg?: number | null
  bmi?: number | null
}

interface SettingsClientProps {
  user: User
  healthProfile?: HealthProfile | null
}

export default function SettingsClient({ user, healthProfile }: SettingsClientProps) {
  const normalizedHealthProfile = healthProfile ? {
    age: healthProfile.age ?? undefined,
    gender: healthProfile.gender ?? undefined,
    medications: healthProfile.medications ?? undefined,
    conditions: healthProfile.conditions ?? undefined,
    allergies: healthProfile.allergies ?? undefined,
    height_cm: healthProfile.height_cm ?? undefined,
    weight_kg: healthProfile.weight_kg ?? undefined,
    bmi: healthProfile.bmi ?? undefined,
  } : undefined
  const [email, setEmail] = useState(user.email || "")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [loadingEmail, setLoadingEmail] = useState(false)
  const [loadingPassword, setLoadingPassword] = useState(false)
  const [loadingDelete, setLoadingDelete] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const callEndpoint = async (path: string, body: any) => {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || "Request failed")
    return data
  }

  const handleEmailUpdate = async (e: React.FormEvent) => {
    e.preventDefault()
    setMessage(null)
    setError(null)
    setLoadingEmail(true)
    try {
      await callEndpoint("/api/account/update-email", { email })
      setMessage("If required, a confirmation link has been sent to your new email address.")
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoadingEmail(false)
    }
  }

  const handlePasswordUpdate = async (e: React.FormEvent) => {
    e.preventDefault()
    setMessage(null)
    setError(null)
    if (newPassword !== confirmPassword) {
      setError("Passwords do not match")
      return
    }
    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters")
      return
    }
    setLoadingPassword(true)
    try {
      await callEndpoint("/api/account/update-password", { password: newPassword })
      setMessage("Password updated successfully")
      setNewPassword("")
      setConfirmPassword("")
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoadingPassword(false)
    }
  }

  const handleDelete = async () => {
    if (!confirm("This will permanently delete your account and data. Continue?")) return
    setMessage(null)
    setError(null)
    setLoadingDelete(true)
    try {
      await callEndpoint("/api/account/delete", {})
      window.location.href = "/auth/login"
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoadingDelete(false)
    }
  }

  return (
  <div className="max-w-4xl mx-auto p-6 space-y-10">
      <h1 className="text-3xl font-semibold tracking-tight text-foreground">Account Settings</h1>
      {error && (
        <Alert className="border border-destructive/50 bg-destructive/15 backdrop-blur-sm">
          <AlertDescription className="text-destructive-foreground/90 font-medium">{error}</AlertDescription>
        </Alert>
      )}
      {message && (
        <Alert className="border border-primary/40 bg-primary/15 backdrop-blur-sm">
          <AlertDescription className="text-primary-foreground/90 font-medium">{message}</AlertDescription>
        </Alert>
      )}

      <Card className="bg-card/60 backdrop-blur border border-border shadow-sm">
        <CardHeader>
          <CardTitle className="text-card-foreground">Update Email</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleEmailUpdate} className="space-y-4">
            <div>
              <label className="block mb-1 text-sm font-medium text-foreground/80">Email</label>
              <Input
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                type="email"
                required
                className="bg-input/40 border-border text-foreground placeholder:text-muted-foreground/60 focus-visible:ring-2 focus-visible:ring-ring"
                disabled={loadingEmail}
              />
            </div>
            <Button type="submit" disabled={loadingEmail} className="shadow-sm">
              {loadingEmail ? "Updating..." : "Update Email"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card className="bg-card/60 backdrop-blur border border-border shadow-sm">
        <CardHeader>
          <CardTitle className="text-card-foreground">Change Password</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handlePasswordUpdate} className="space-y-4">
            <div>
              <label className="block mb-1 text-sm font-medium text-foreground/80">New Password</label>
              <Input
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                type="password"
                required
                className="bg-input/40 border-border text-foreground placeholder:text-muted-foreground/60 focus-visible:ring-2 focus-visible:ring-ring"
                disabled={loadingPassword}
              />
            </div>
            <div>
              <label className="block mb-1 text-sm font-medium text-foreground/80">Confirm Password</label>
              <Input
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                type="password"
                required
                className="bg-input/40 border-border text-foreground placeholder:text-muted-foreground/60 focus-visible:ring-2 focus-visible:ring-ring"
                disabled={loadingPassword}
              />
            </div>
            <Button type="submit" disabled={loadingPassword} className="shadow-sm">
              {loadingPassword ? "Updating..." : "Change Password"}
            </Button>
          </form>
        </CardContent>
      </Card>

  <HealthProfileForm initialProfile={normalizedHealthProfile} />

  <Card className="bg-destructive/10 border border-destructive/40 shadow-sm">
        <CardHeader>
          <CardTitle className="text-destructive tracking-wide">Danger Zone</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">Permanently delete your account and all associated data.</p>
          <Button
            type="button"
            variant="destructive"
            disabled={loadingDelete}
            onClick={handleDelete}
            className="shadow-sm"
          >
            {loadingDelete ? "Deleting..." : "Delete Account"}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
