import { NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"

export async function POST(request: Request) {
  try {
    const { password } = await request.json()
    if (!password || password.length < 6) return NextResponse.json({ error: "Password too short" }, { status: 400 })
    const supabase = await createClient()
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser()
    if (authError || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 })

    const { error } = await supabase.auth.updateUser({ password })
    if (error) return NextResponse.json({ error: error.message }, { status: 400 })

    return NextResponse.json({ success: true })
  } catch (e) {
    return NextResponse.json({ error: "Server error" }, { status: 500 })
  }
}
