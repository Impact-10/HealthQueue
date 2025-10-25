import { redirect } from "next/navigation"
import { createClient } from "@/lib/supabase/server"
import { DashboardLayout } from "@/components/dashboard/dashboard-layout"

export const metadata = { title: 'Dashboard' }

export default async function DashboardPage() {
  const supabase = await createClient()

  const {
    data: { user },
    error,
  } = await supabase.auth.getUser()

  console.log("[v0] Dashboard access attempt:", { user: user?.id, error })

  if (error || !user) {
    console.log("[v0] Redirecting to login - no valid user session")
    redirect("/auth/login")
  }

  try {
    let { data: profile, error: profileError } = await supabase.from("profiles").select("*").eq("id", user.id).single()

    if (profileError && profileError.code === "PGRST116") {
      // Profile doesn't exist, create it
      console.log("[v0] Creating new profile for user:", user.id)
      const { data: newProfile, error: createError } = await supabase
        .from("profiles")
        .insert({
          id: user.id,
          email: user.email,
          full_name: user.user_metadata?.full_name || null,
          avatar_url: user.user_metadata?.avatar_url || null,
        })
        .select()
        .single()

      if (createError) {
        console.log("[v0] Profile creation error:", createError)
        profile = null
      } else {
        profile = newProfile
      }
    } else if (profileError) {
      console.log("[v0] Profile fetch error:", profileError)
      profile = null
    }

    console.log("[v0] Dashboard loaded successfully for user:", user.id)
    return <DashboardLayout user={user} profile={profile} />
  } catch (error) {
    console.log("[v0] Dashboard error:", error)
    // Still render dashboard even if profile operations fail
    return <DashboardLayout user={user} profile={null} />
  }
}
