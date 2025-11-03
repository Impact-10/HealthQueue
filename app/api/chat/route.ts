/* eslint-disable @typescript-eslint/no-explicit-any */
import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

const BACKEND_URL = process.env.DIAGNOSAI_BACKEND_URL || "http://localhost:8000"
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)

const deriveStructuredSummary = (structured: unknown): string => {
  if (!structured || typeof structured !== "object") return ""
  const c = (structured as any).content
  if (c && typeof c === "object") {
    for (const k of ["summary","answer","text","raw_text"]) {
      const v = (c as any)[k]; if (typeof v === "string" && v.trim()) return v.trim()
    }
    for (const k of ["possible_causes","possible_diagnoses","recommendations","next_steps","warning_signs"]) {
      const v = (c as any)[k]; if (Array.isArray(v) && v.length) return v.slice(0,3).map(String).join(", ")
    }
  }
  return ""
}

const synthesizeIfEmpty = (structured: any): string => {
  try {
    const c = structured?.content || {}
    const s = [c.summary, c.text, c.answer, c.raw_text].find((v:any)=> typeof v==="string" && v.trim())
    if (s) return String(s).trim()
    for (const k of ["recommendations","possible_causes","next_steps","warning_signs","recommended_tests","possible_diagnoses"]) {
      const arr = c[k]; if (Array.isArray(arr) && arr.length) return String(arr[0])
    }
  } catch {}
  return ""
}

export async function POST(request: NextRequest) {
  try {
    const { message, fileUrls, threadId, userId, modelKey } = await request.json()
    const supabase = await createClient()

    const { data: { user } } = await supabase.auth.getUser()
    if (!user || user.id !== userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 })

    const { data: usageData } = await supabase.rpc("check_usage_limit", { p_user_id: userId, p_daily_message_limit: 50, p_daily_file_limit: 10 })
    if (!usageData?.can_send_message) return NextResponse.json({ error: "Daily message limit reached" }, { status: 429 })

    const fetchWithTimeout = async (url: string, body: any) => {
      const controller = new AbortController()
      const tm = setTimeout(()=>controller.abort(), 20000)
      const resp = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: controller.signal }).catch((e)=>{ clearTimeout(tm); throw e })
      clearTimeout(tm)
      return resp
    }

    const persist = async (tid?: string|null, text?: string|null, structured?: any, direct?: string|null, modelInfo?: any) => {
      if (!tid) {
        const { data: newThread } = await supabase.from("threads").insert({ user_id: userId, title: message.slice(0,50) + (message.length>50?"...":"") }).select().single()
        tid = newThread!.id
      }
      await supabase.from("messages").insert({ thread_id: tid, user_id: userId, role: "user", content: message, file_urls: fileUrls }).select().single()
      const stored = structured && typeof structured === "object"
        ? { type: "structured-response", version: 1, structuredResponse: structured, text: text ?? undefined, modelKey, directAnswer: direct ?? undefined, modelInfo: modelInfo ?? undefined, createdAt: new Date().toISOString() }
        : null
      const content = stored ? JSON.stringify(stored) : (text || "")
      const { data: assistantMessage } = await supabase.from("messages").insert({ thread_id: tid, user_id: userId, role: "assistant", content }).select().single()
      await supabase.rpc("update_usage_tracking", { p_user_id: userId, p_message_count: 1, p_file_upload_count: fileUrls?.length || 0 })
      return { tid, msgId: assistantMessage!.id }
    }

    if (modelKey === "biogpt") {
      const resp = await fetchWithTimeout(`${BACKEND_URL}/api/biogpt`, { question: message })
      if (!resp.ok) {
        const errTxt = await resp.text(); console.error("[BioGPT API] Backend error:", errTxt)
        return NextResponse.json({ error: "Backend unavailable" }, { status: 502 })
      }
      let backendJson: any
      try { backendJson = await resp.json() } catch { const raw = await resp.text(); return NextResponse.json({ error: `Backend error: ${raw}` }, { status: 502 }) }
      const structuredResponse = backendJson?.structuredResponse ?? backendJson
      const modelInfo = backendJson?.model ?? null
      const directAnswer = (typeof backendJson?.directAnswer === "string" && backendJson.directAnswer.trim()) ? backendJson.directAnswer.trim() : null
      const fallbackText = deriveStructuredSummary(structuredResponse)
      let responseText = directAnswer ?? (fallbackText || null) ?? (typeof backendJson?.response === "string" ? backendJson.response : null)
      if (!responseText || !responseText.trim()) responseText = synthesizeIfEmpty(structuredResponse) || "Let’s consider common causes and next steps."
      const saved = await persist(threadId, responseText, structuredResponse, directAnswer, modelInfo)
      return NextResponse.json({ structuredResponse, directAnswer, modelInfo, response: responseText, threadId: saved.tid, messageId: saved.msgId })
    }

    if (modelKey === "medalpaca") {
      const resp = await fetchWithTimeout(`${BACKEND_URL}/api/medalpaca`, { question: message })
      if (!resp.ok) {
        const errTxt = await resp.text(); console.error("[MedAlpaca API] Backend error:", errTxt)
        return NextResponse.json({ error: "Backend unavailable" }, { status: 502 })
      }
      let backendJson: any
      try { backendJson = await resp.json() } catch { const raw = await resp.text(); return NextResponse.json({ error: `Backend error: ${raw}` }, { status: 502 }) }
      const structuredResponse = backendJson?.structuredResponse ?? backendJson
      const modelInfo = backendJson?.model ?? null
      const directAnswer = (typeof backendJson?.directAnswer === "string" && backendJson.directAnswer.trim()) ? backendJson.directAnswer.trim() : null
      const fallbackText = deriveStructuredSummary(structuredResponse)
      let responseText = directAnswer ?? (fallbackText || null) ?? (typeof backendJson?.response === "string" ? backendJson.response : null)
      if (!responseText || !responseText.trim()) responseText = synthesizeIfEmpty(structuredResponse) || "Here’s a concise health note based on your message."
      const saved = await persist(threadId, responseText, structuredResponse, directAnswer, modelInfo)
      return NextResponse.json({ structuredResponse, directAnswer, modelInfo, response: responseText, threadId: saved.tid, messageId: saved.msgId })
    }

    // Gemini fallback for any other key
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" })
    const result = await model.generateContent(`You are a health assistant.\n\nUser: ${message}`)
    const aiResponse = result.response.text()
    let tid = threadId
    if (!tid) {
      const { data: newThread } = await supabase.from("threads").insert({ user_id: userId, title: message.slice(0,50) + (message.length>50?"...":"") }).select().single()
      tid = newThread!.id
    }
    await supabase.from("messages").insert({ thread_id: tid, user_id: userId, role: "user", content: message, file_urls: fileUrls }).select().single()
    const { data: assistantMessage } = await supabase.from("messages").insert({ thread_id: tid, user_id: userId, role: "assistant", content: aiResponse }).select().single()
    await supabase.rpc("update_usage_tracking", { p_user_id: userId, p_message_count: 1, p_file_upload_count: fileUrls?.length || 0 })
    return NextResponse.json({ response: aiResponse, threadId: tid, messageId: assistantMessage!.id })
  } catch (e) {
    console.error("[/api/chat] Unexpected error:", e)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
