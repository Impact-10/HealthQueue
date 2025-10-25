import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)

const HEALTH_SYSTEM_PROMPT = `You are an AI Health Assistant, a knowledgeable and empathetic healthcare companion. You must ONLY engage on health, wellness, fitness, nutrition, mental health support, preventive care, medical literacy, and related lifestyle guidance. Politely refuse and redirect any request outside these domains. Your role is to:

1. Provide accurate, evidence-based health information
2. Help users understand medical concepts and terminology
3. Suggest when to seek professional medical care
4. Support users in tracking their health journey
5. Offer wellness and lifestyle guidance

IMPORTANT GUIDELINES:
- Always emphasize that you are not a replacement for professional medical advice
- Encourage users to consult healthcare providers for serious concerns
- Be empathetic and supportive in your responses
- Ask clarifying questions when needed
- Provide actionable, practical advice when appropriate
- Respect user privacy and maintain confidentiality

DISALLOWED / OUT-OF-SCOPE TOPICS:
- General chit-chat unrelated to health (e.g., sports scores, movie plots, coding help)
- Political opinions, legal advice, financial investing, gambling, hacking, or explicit content
- Creative writing unrelated to health education
- Personal data extraction beyond what the user already shared

If a user asks anything outside scope, respond briefly: "This question isn't related to health or wellness. Please ask a health-related question so I can help you." Do not provide partial answers to out-of-scope topics.

SAFETY PROTOCOLS:
- Never provide specific medical diagnoses
- Always recommend professional consultation for serious symptoms
- Be cautious with medication advice
- Encourage emergency care for urgent situations

Remember: You are a supportive companion in the user's health journey, not a medical professional.`

export async function POST(request: NextRequest) {
  try {
    const { message, fileUrls, threadId, userId } = await request.json()

    if (!message && (!fileUrls || fileUrls.length === 0)) {
      return NextResponse.json({ error: "Message or files required" }, { status: 400 })
    }

    // Basic topic filtering (fast heuristic). Could be expanded with an embedding or classifier.
    const LOWER = (message || "").toLowerCase()
    const HEALTH_KEYWORDS = [
      "health",
      "wellness",
      "nutrition",
      "diet",
      "exercise",
      "fitness",
      "workout",
      "sleep",
      "stress",
      "anxiety",
      "mental",
      "symptom",
      "medicine",
      "medication",
      "blood",
      "heart",
      "cholesterol",
      "diabetes",
      "hypertension",
      "injury",
      "recovery",
      "therapy",
      "clinical",
      "medical",
      "allergy",
      "allergies",
      "vitamin",
      "supplement",
      "immunity",
      "infection",
      "disease",
      "condition",
      "diagnosis",
      "treatment",
      "doctor",
      "nurse",
      "hospital",
      "clinic",
      "surgery",
      "pregnancy",
      "birth",
      "childbirth",
      "pediatric",
      "geriatrics",
      "cancer",
      "tumor",
      "mental health",
      "depression",
      "bipolar",
      "schizophrenia",
      "ptsd",
      "adhd",
      "autism",
      "covid",
      "pandemic",
      "flu",
      "influenza",
      "vaccine",
      "vaccination",
      "public health",
      "hygiene",
      "sanitation",
      "first aid",
      "emergency",
      "cpR",
      "aed",
      "wound",
      "burn",
      "fracture",
      "sprain",
      "strain",
      "headache",
      "migraine",
      "fever",
      "cough",
      "cold",
      "flu",
      "stomach ache",
      "diarrhea",
      "constipation",
      "nausea",
      "vomiting",
      "dizziness",
      "fatigue",
      "chronic pain",
      "arthritis",
      "asthma",
      "allergic reaction",
      "anaphylaxis",
      "mental illness",
      "substance abuse",
      "addiction",
      "rehabilitation",
      "detoxification",
      "harm reduction",
      "self-care",
      "mindfulness",
      "meditation",
      "yoga"
    ]

    const isHealthLike = HEALTH_KEYWORDS.some((k) => LOWER.includes(k))

    if (!isHealthLike) {
      const refusal = "This question isn't related to health or wellness. Please ask a health-related question so I can help you."
      return NextResponse.json({
        response: refusal,
        threadId: threadId || null,
        messageId: null,
        filtered: true,
      })
    }

    const supabase = await createClient()

    // Verify user authentication
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser()

    if (authError || !user || user.id !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Check usage limits
    const { data: usageData } = await supabase.rpc("check_usage_limit", {
      p_user_id: userId,
      p_daily_message_limit: 50,
      p_daily_file_limit: 10,
    })

    if (!usageData?.can_send_message) {
      return NextResponse.json({ error: "Daily message limit reached" }, { status: 429 })
    }

    // Get user profile for context
    const { data: profile } = await supabase.from("profiles").select("*").eq("id", userId).single()

    // Get recent conversation history for context
    let conversationHistory = ""
    if (threadId) {
      const { data: recentMessages } = await supabase
        .from("messages")
        .select("role, content")
        .eq("thread_id", threadId)
        .order("created_at", { ascending: false })
        .limit(10)

      if (recentMessages && recentMessages.length > 0) {
        conversationHistory = recentMessages
          .reverse()
          .map((msg) => `${msg.role}: ${msg.content}`)
          .join("\n")
      }
    }

    // Build context for AI
    let contextualPrompt = HEALTH_SYSTEM_PROMPT

    if (profile) {
      contextualPrompt += `\n\nUser Profile Context:
- Age: ${profile.date_of_birth ? calculateAge(profile.date_of_birth) : "Not provided"}
- Gender: ${profile.gender || "Not provided"}
- Activity Level: ${profile.activity_level || "Not provided"}
- Medical Conditions: ${profile.medical_conditions?.join(", ") || "None reported"}
- Medications: ${profile.medications?.join(", ") || "None reported"}
- Allergies: ${profile.allergies?.join(", ") || "None reported"}
- Health Goals: ${profile.health_goals?.join(", ") || "Not specified"}`
    }

    if (conversationHistory) {
      contextualPrompt += `\n\nRecent Conversation History:\n${conversationHistory}`
    }

    // Process file content if provided
    let fileContent = ""
    if (fileUrls && fileUrls.length > 0) {
      fileContent =
        "\n\nUser has attached files. Please acknowledge the files and ask how you can help analyze or discuss them."
    }

    // Generate AI response
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" })

    const fullPrompt = `${contextualPrompt}\n\nUser Message: ${message}${fileContent}`

    const result = await model.generateContent(fullPrompt)
    const aiResponse = result.response.text()

    // Create or get thread
    let currentThreadId = threadId
    if (!currentThreadId) {
      const { data: newThread, error: threadError } = await supabase
        .from("threads")
        .insert({
          user_id: userId,
          title: message.slice(0, 50) + (message.length > 50 ? "..." : ""),
        })
        .select()
        .single()

      if (threadError) throw threadError
      currentThreadId = newThread.id
    }

    // Save user message
    const { data: userMessage, error: userMessageError } = await supabase
      .from("messages")
      .insert({
        thread_id: currentThreadId,
        user_id: userId,
        role: "user",
        content: message,
        file_urls: fileUrls,
      })
      .select()
      .single()

    if (userMessageError) throw userMessageError

    // Save AI response
    const { data: assistantMessage, error: assistantMessageError } = await supabase
      .from("messages")
      .insert({
        thread_id: currentThreadId,
        user_id: userId,
        role: "assistant",
        content: aiResponse,
      })
      .select()
      .single()

    if (assistantMessageError) throw assistantMessageError

    // Update usage tracking
    await supabase.rpc("update_usage_tracking", {
      p_user_id: userId,
      p_message_count: 1,
      p_file_upload_count: fileUrls?.length || 0,
    })

    return NextResponse.json({
      response: aiResponse,
      threadId: currentThreadId,
      messageId: assistantMessage.id,
    })
  } catch (error) {
    console.error("Chat API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

function calculateAge(birthDate: string): number {
  const today = new Date()
  const birth = new Date(birthDate)
  let age = today.getFullYear() - birth.getFullYear()
  const monthDiff = today.getMonth() - birth.getMonth()

  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age--
  }

  return age
}
