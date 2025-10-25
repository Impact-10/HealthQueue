import { NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)

const REPORT_SYSTEM_PROMPT = `You are a medical summary assistant. Create a concise, clinically useful encounter-style report for a licensed healthcare professional based on the patient's chat thread. Follow this structure:

Patient Summary:
- Age / sex (if available). Key medical history and relevant context.

Presenting Concerns:
- Bullet list of main symptoms / questions.

History Details:
- Chronology and pertinent positives / negatives.

Current Medications & Allergies:
- Summarize if mentioned; otherwise state not provided.

Lifestyle Factors:
- Diet, exercise, sleep, stress, substance use (only if mentioned).

Assessment (Non-Diagnostic):
- Neutral restatement of possible categories (avoid definitive diagnoses). Avoid speculation beyond provided info.

Red Flags / Urgent Considerations:
- List any symptoms that warrant urgent evaluation or 'None noted'.

Recommended Next Steps (General Guidance Only):
- Practical self-care, tracking, or questions to ask a doctor. Avoid prescribing medications.

Disclaimers:
- This summary is informational and not a diagnosis. Encourage professional evaluation.

Rules:
- DO NOT fabricate details.
- If data is missing, explicitly note 'not mentioned'.
- Keep to < 350 words if possible.
- Use clear bullet formatting.
`

export async function POST(req: NextRequest) {
  try {
    const { threadId, userId } = await req.json()
    if (!threadId || !userId) {
      return NextResponse.json({ error: "threadId and userId required" }, { status: 400 })
    }

    const supabase = await createClient()
    const { data: { user } } = await supabase.auth.getUser()
    if (!user || user.id !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Fetch messages for the thread
    const { data: msgs, error: msgErr } = await supabase
      .from("messages")
      .select("role, content, created_at")
      .eq("thread_id", threadId)
      .order("created_at", { ascending: true })
    if (msgErr) throw msgErr
    if (!msgs || msgs.length === 0) {
      return NextResponse.json({ error: "No messages in thread" }, { status: 404 })
    }

    // Fetch profile if present
    const { data: profile } = await supabase.from("profiles").select("date_of_birth, gender, medical_conditions, medications, allergies, activity_level, health_goals").eq("id", userId).single()

    const age = profile?.date_of_birth ? calculateAge(profile.date_of_birth) : null

    const conversationText = msgs
      .map(m => `${m.role === 'assistant' ? 'Assistant' : 'Patient'}: ${m.content}`)
      .join("\n")

    const profileContext = `Patient Context:\n- Age: ${age ?? 'not mentioned'}\n- Gender: ${profile?.gender || 'not mentioned'}\n- Conditions: ${profile?.medical_conditions?.join(', ') || 'not mentioned'}\n- Medications: ${profile?.medications?.join(', ') || 'not mentioned'}\n- Allergies: ${profile?.allergies?.join(', ') || 'not mentioned'}\n- Activity Level: ${profile?.activity_level || 'not mentioned'}\n- Goals: ${profile?.health_goals?.join(', ') || 'not mentioned'}`

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" })
    const fullPrompt = `${REPORT_SYSTEM_PROMPT}\n\n${profileContext}\n\nConversation Transcript:\n${conversationText}`
    const result = await model.generateContent(fullPrompt)
    const report = result.response.text()

    return NextResponse.json({ report })
  } catch (e) {
    console.error('Report generation error', e)
    return NextResponse.json({ error: "Failed to generate report" }, { status: 500 })
  }
}

function calculateAge(birthDate: string): number {
  const today = new Date()
  const birth = new Date(birthDate)
  let age = today.getFullYear() - birth.getFullYear()
  const monthDiff = today.getMonth() - birth.getMonth()
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) age--
  return age
}
