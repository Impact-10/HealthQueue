import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import SettingsClient from "./settings-client"

export const dynamic = "force-dynamic"
export const metadata = { title: 'Settings' }

export default async function SettingsPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth/login")
  }

  // Fetch patient profile (may not exist; swallow single-row missing error)
  let healthProfile: any = null
  try {
    const { data, error } = await supabase
      .from('patient_profiles')
      .select('age, gender, medications, conditions, allergies, height_cm, weight_kg, bmi')
      .eq('user_id', user.id)
      .single()
    if (!error) healthProfile = data
  } catch {}

  return <SettingsClient user={user} healthProfile={healthProfile} />
}
