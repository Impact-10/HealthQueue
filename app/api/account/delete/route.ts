import { NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"

// NOTE: Requires a Postgres function / RLS policy or service role for full deletion if needed.
export async function POST() {
  try {
    const supabase = await createClient()
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser()
    if (authError || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 })

    // Delete related data (threads, messages, profile) - rely on FK cascade or explicit deletes
    await supabase.from("messages").delete().eq("user_id", user.id)
    await supabase.from("threads").delete().eq("user_id", user.id)
    await supabase.from("profiles").delete().eq("id", user.id)

    // Supabase client anon key cannot delete auth user directly (needs service role). Instead, sign them out.
    await supabase.auth.signOut()

    return NextResponse.json({ success: true })
  } catch (e) {
    return NextResponse.json({ error: "Server error" }, { status: 500 })
  }
}
