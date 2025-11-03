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

    // Auth: prefer session user to avoid mismatch 500s during dev
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    const authedUserId = user.id
    if (userId && userId !== authedUserId) {
      // In prod you can enforce equality; for now surface clearly
      return NextResponse.json({ error: "Unauthorized (user mismatch)" }, { status: 401 })
    }

    // Usage limits
    const { data: usageData, error: usageErr } = await supabase.rpc("check_usage_limit", { p_user_id: authedUserId, p_daily_message_limit: 50, p_daily_file_limit: 10 })
    if (usageErr) {
      console.error("[/api/chat] usage check error:", usageErr)
      return NextResponse.json({ error: "Usage check failed" }, { status: 500 })
    }
    if (!usageData?.can_send_message) return NextResponse.json({ error: "Daily message limit reached" }, { status: 429 })

    // Backend fetch with longer timeout and clearer error
    const fetchWithTimeout = async (url: string, body: any) => {
      const controller = new AbortController()
      const tm = setTimeout(()=>controller.abort(), 45000) // 45s to avoid cold-start aborts
      try {
        const resp = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal
        })
        clearTimeout(tm)
        return resp
      } catch (e: any) {
        clearTimeout(tm)
        if (e?.name === "AbortError") throw new Error("Backend fetch timed out")
        throw e
      }
    }

    const persist = async (tid?: string|null, text?: string|null, structured?: any, direct?: string|null, modelInfo?: any) => {
      let thread = tid
      if (!thread) {
        const { data: newThread, error: tErr } = await supabase
          .from("threads")
          .insert({ user_id: authedUserId, title: message.slice(0,50) + (message.length>50?"...":"") })
          .select()
          .single()
        if (tErr) throw new Error("Failed to create thread")
        thread = newThread!.id
      }
      const uIns = await supabase
        .from("messages")
        .insert({ thread_id: thread, user_id: authedUserId, role: "user", content: message, file_urls: fileUrls })
        .select()
        .single()
      if (uIns.error) throw new Error("Failed to store user message")

      const stored = structured && typeof structured === "object"
        ? { type: "structured-response", version: 1, structuredResponse: structured, text: text ?? undefined, modelKey, directAnswer: direct ?? undefined, modelInfo: modelInfo ?? undefined, createdAt: new Date().toISOString() }
        : null
      const content = stored ? JSON.stringify(stored) : (text || "")
      const aIns = await supabase
        .from("messages")
        .insert({ thread_id: thread, user_id: authedUserId, role: "assistant", content })
        .select()
        .single()
      if (aIns.error) throw new Error("Failed to store assistant message")

      await supabase.rpc("update_usage_tracking", { p_user_id: authedUserId, p_message_count: 1, p_file_upload_count: fileUrls?.length || 0 })
      return { tid: thread, msgId: aIns.data!.id }
    }

    if (modelKey === "biogpt") {
      let resp: Response
      try {
        resp = await fetchWithTimeout(`${BACKEND_URL}/api/biogpt`, { question: message })
      } catch (e:any) {
        console.error("[BioGPT API] fetch failed:", e)
        return NextResponse.json({ error: e?.message || "Backend fetch failed" }, { status: 502 })
      }
      if (!resp.ok) {
        const errTxt = await resp.text()
        console.error("[BioGPT API] Backend error:", errTxt)
        return NextResponse.json({ error: `Backend error: ${errTxt}` }, { status: 502 })
      }
      let backendJson: any
      try {
        backendJson = await resp.json()
      } catch {
        const raw = await resp.text()
        return NextResponse.json({ error: `Backend error: ${raw}` }, { status: 502 })
      }
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
      let resp: Response
      try {
        resp = await fetchWithTimeout(`${BACKEND_URL}/api/medalpaca`, { question: message })
      } catch (e:any) {
        console.error("[MedAlpaca API] fetch failed:", e)
        return NextResponse.json({ error: e?.message || "Backend fetch failed" }, { status: 502 })
      }
      if (!resp.ok) {
        const errTxt = await resp.text()
        console.error("[MedAlpaca API] Backend error:", errTxt)
        return NextResponse.json({ error: `Backend error: ${errTxt}` }, { status: 502 })
      }
      let backendJson: any
      try {
        backendJson = await resp.json()
      } catch {
        const raw = await resp.text()
        return NextResponse.json({ error: `Backend error: ${raw}` }, { status: 502 })
      }
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
      const { data: newThread, error: tErr } = await supabase
        .from("threads")
        .insert({ user_id: authedUserId, title: message.slice(0,50) + (message.length>50?"...":"") })
        .select()
        .single()
      if (tErr) return NextResponse.json({ error: "Failed to create thread" }, { status: 500 })
      tid = newThread!.id
    }
    const uIns = await supabase
      .from("messages")
      .insert({ thread_id: tid, user_id: authedUserId, role: "user", content: message, file_urls: fileUrls })
      .select()
      .single()
    if (uIns.error) return NextResponse.json({ error: "Failed to store user message" }, { status: 500 })
    const aIns = await supabase
      .from("messages")
      .insert({ thread_id: tid, user_id: authedUserId, role: "assistant", content: aiResponse })
      .select()
      .single()
    if (aIns.error) return NextResponse.json({ error: "Failed to store assistant message" }, { status: 500 })
    await supabase.rpc("update_usage_tracking", { p_user_id: authedUserId, p_message_count: 1, p_file_upload_count: fileUrls?.length || 0 })
    return NextResponse.json({ response: aiResponse, threadId: tid, messageId: aIns.data!.id })

  } catch (e: any) {
    console.error("[/api/chat] Unexpected error:", e)
    const msg = typeof e?.message === "string" ? e.message : "Internal server error"
    return NextResponse.json({ error: msg }, { status: 500 })
  }
}
