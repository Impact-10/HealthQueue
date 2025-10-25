import { NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"

export async function POST(req: NextRequest) {
  try {
    const { sourceThreadId, userId, doctorId } = await req.json()
    if (!userId || !sourceThreadId) {
      return NextResponse.json({ error: "userId and sourceThreadId required" }, { status: 400 })
    }
    const supabase = await createClient()
    const { data: { user } } = await supabase.auth.getUser()
    if (!user || user.id !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Fetch existing report (reuse /api/report logic by inlining minimal version here for efficiency)
    const { data: msgs } = await supabase
      .from("messages")
      .select("role, content, created_at")
      .eq("thread_id", sourceThreadId)
      .order("created_at", { ascending: true })
    if (!msgs || msgs.length === 0) {
      return NextResponse.json({ error: "Source thread empty" }, { status: 400 })
    }

    // Basic derived title from first user message
    const firstUser = msgs.find(m => m.role === 'user')
    const title = (firstUser?.content || 'Consultation').slice(0, 60)

    // Attempt to call report API internally (simpler fetch for code reuse)
    let report: string | null = null
    try {
      const origin = process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000'
      const res = await fetch(`${origin}/api/report`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ threadId: sourceThreadId, userId }) })
      if (res.ok) {
        const data = await res.json(); report = data.report
      }
    } catch (e) {
      // swallow, fallback to transcript embed
    }

    // (Future) Fetch patient profile (age, medications, conditions) if table exists
    let profileSummary: string | null = null
    try {
      const { data: p } = await supabase
        .from('patient_profiles')
        .select('age, gender, medications, conditions, allergies, height_cm, weight_kg, bmi')
        .eq('user_id', userId)
        .single()
      if (p) {
        const metrics = [] as string[]
        if (p.height_cm) metrics.push(`Height: ${p.height_cm} cm`)
        if (p.weight_kg) metrics.push(`Weight: ${p.weight_kg} kg`)
        if (p.bmi) metrics.push(`BMI: ${p.bmi}`)
        profileSummary = `Patient Profile:\n- Age: ${p.age ?? 'N/A'}\n- Gender: ${p.gender || 'N/A'}\n- ${metrics.join(' | ') || 'No anthropometrics'}\n- Medications: ${p.medications || 'None reported'}\n- Conditions: ${p.conditions || 'None reported'}\n- Allergies: ${p.allergies || 'None reported'}`
      }
    } catch (e) {
      // swallow profile errors (table may not exist yet)
    }

    // Create doctor thread
    const { data: newThread, error: dtErr } = await supabase
      .from('doctor_threads')
      .insert({ user_id: userId, doctor_id: doctorId || null, source_thread_id: sourceThreadId, title })
      .select()
      .single()
    if (dtErr) throw dtErr

    const systemIntro = `System: This consultation thread was initiated by the patient. The following report summarizes the patient's AI assistant conversation and context.`
    const summaryContent = (report || 'Report generation failed; including raw recent transcript.\n\n' + msgs.slice(-10).map(m => `${m.role}: ${m.content}`).join('\n')) + (profileSummary ? ('\n\n' + profileSummary) : '')
    const doctorGreeting = `doctor: Hello, I'm reviewing your provided summary now. I'll ask a couple of clarifying questions to ensure I understand your situation correctly. Feel free to add anything you think is important.`

    // Seed system + report messages (as system role, reuse doctor_messages table with role system/user distinction)
    const seedMessages = [
      { doctor_thread_id: newThread.id, role: 'system', content: systemIntro },
      { doctor_thread_id: newThread.id, role: 'system', content: summaryContent },
      { doctor_thread_id: newThread.id, role: 'doctor', content: doctorGreeting },
    ]
    const { error: insErr } = await supabase.from('doctor_messages').insert(seedMessages)
    if (insErr) throw insErr

    return NextResponse.json({ doctorThreadId: newThread.id })
  } catch (e) {
    console.error('Doctor thread creation error', e)
    return NextResponse.json({ error: 'Failed to create doctor thread' }, { status: 500 })
  }
}
