

import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

const BACKEND_URL = process.env.DIAGNOSAI_BACKEND_URL || "http://localhost:8000"

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)

const STRUCTURED_PAYLOAD_TYPE = "structured-response"
const STRUCTURED_PAYLOAD_VERSION = 1

const deriveStructuredSummary = (structured: unknown): string => {
  if (!structured || typeof structured !== "object") return ""

  const container = structured as Record<string, unknown>
  const content = container.content

  if (content && typeof content === "object" && !Array.isArray(content)) {
    const contentRecord = content as Record<string, unknown>
    const preferredKeys = ["summary", "answer", "text", "raw_text"]
    for (const key of preferredKeys) {
      const value = contentRecord[key]
      if (typeof value === "string" && value.trim().length > 0) {
        return value.trim()
      }
    }

    const listKeys = [
      "possible_causes",
      "possible_diagnoses",
      "recommendations",
      "next_steps",
      "warning_signs",
    ]
    for (const key of listKeys) {
      const value = contentRecord[key]
      if (Array.isArray(value) && value.length > 0) {
        return value.slice(0, 3).map((item) => String(item)).join(", ")
      }
    }

    if ("entities" in contentRecord) {
      const entities = contentRecord.entities
      if (entities && typeof entities === "object" && !Array.isArray(entities)) {
        const segments: string[] = []
        for (const [etype, values] of Object.entries(entities as Record<string, unknown>)) {
          if (Array.isArray(values) && values.length > 0) {
            segments.push(`${etype}: ${values.slice(0, 2).map((item) => String(item)).join(", ")}`)
          }
        }
        if (segments.length > 0) {
          return segments.join(" | ")
        }
      }
    }

    if ("individual_results" in contentRecord) {
      const individual = contentRecord.individual_results
      if (individual && typeof individual === "object" && !Array.isArray(individual)) {
        const keys = Object.keys(individual as Record<string, unknown>)
        if (keys.length > 0) {
          return `Responses from ${keys.length} model${keys.length > 1 ? "s" : ""}`
        }
      }
    }
  }

  if (typeof container.model === "string" && container.model.trim().length > 0) {
    return container.model.trim()
  }

  return ""
}

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
    // Parse request body
    const { message, fileUrls, threadId, userId, modelKey } = await request.json();
    const supabase = await createClient();

    // Authenticate user
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();
    if (authError || !user || user.id !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Check usage limits
    const { data: usageData } = await supabase.rpc("check_usage_limit", {
      p_user_id: userId,
      p_daily_message_limit: 50,
      p_daily_file_limit: 10,
    });
    if (!usageData?.can_send_message) {
      return NextResponse.json({ error: "Daily message limit reached" }, { status: 429 });
    }

    // Get user profile for context
    const { data: profile } = await supabase.from("profiles").select("*").eq("id", userId).single();

    // Get recent conversation history for context
    let conversationHistory = "";
    if (threadId) {
      const { data: recentMessages } = await supabase
        .from("messages")
        .select("role, content")
        .eq("thread_id", threadId)
        .order("created_at", { ascending: false })
        .limit(10);
      if (recentMessages && recentMessages.length > 0) {
        conversationHistory = recentMessages
          .reverse()
          .map((msg: { role: string; content: string }) => `${msg.role}: ${msg.content}`)
          .join("\n");
      }
    }

    // Build context for AI
    let contextualPrompt = HEALTH_SYSTEM_PROMPT;
    if (profile) {
      contextualPrompt += `\n\nUser Profile Context:
- Age: ${profile.date_of_birth ? calculateAge(profile.date_of_birth) : "Not provided"}
- Gender: ${profile.gender || "Not provided"}
- Activity Level: ${profile.activity_level || "Not provided"}
- Medical Conditions: ${profile.medical_conditions?.join(", ") || "None reported"}
- Medications: ${profile.medications?.join(", ") || "None reported"}
- Allergies: ${profile.allergies?.join(", ") || "None reported"}
- Health Goals: ${profile.health_goals?.join(", ") || "Not specified"}`;
    }
    if (conversationHistory) {
      contextualPrompt += `\n\nRecent Conversation History:\n${conversationHistory}`;
    }

    // Process file content if provided
    let fileContent = "";
    if (fileUrls && fileUrls.length > 0) {
      fileContent =
        "\n\nUser has attached files. Please acknowledge the files and ask how you can help analyze or discuss them.";
    }

    // --- Only MedAlpaca and Gemini model routing ---
    if (modelKey === "medalpaca") {
      try {
        const backendResp = await fetch(`${BACKEND_URL}/api/medalpaca`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: message }),
        });
        if (!backendResp.ok) {
          const err = await backendResp.text();
          console.error(`[MedAlpaca API] Backend error:`, err);
          return NextResponse.json({ error: `Backend error: ${typeof err === 'string' ? err : JSON.stringify(err)}` }, { status: 502 });
        }
        const backendJson = await backendResp.json();
        const structuredResponse = backendJson?.structuredResponse ?? backendJson;
        const modelInfo = backendJson?.model ?? null;
        const directAnswerCandidate = typeof backendJson?.directAnswer === "string"
          ? backendJson.directAnswer
          : typeof backendJson?.answer === "string"
            ? backendJson.answer
            : null;
        const directAnswer = directAnswerCandidate && directAnswerCandidate.trim().length > 0
          ? directAnswerCandidate.trim()
          : null;
        const fallbackText = deriveStructuredSummary(structuredResponse);
        const responseText = directAnswer
          ?? (fallbackText && fallbackText.length > 0 ? fallbackText : null)
          ?? (typeof backendJson?.response === "string" ? backendJson.response : null);
        // Create or get thread
        let currentThreadId = threadId;
        if (!currentThreadId) {
          const { data: newThread, error: threadError } = await supabase
            .from("threads")
            .insert({
              user_id: userId,
              title: message.slice(0, 50) + (message.length > 50 ? "..." : ""),
            })
            .select()
            .single();
          if (threadError) {
            console.error(`[MedAlpaca API] Thread creation error:`, threadError);
            throw threadError;
          }
          currentThreadId = newThread.id;
        }
        // Save user message
        const { error: userMessageError } = await supabase
          .from("messages")
          .insert({
            thread_id: currentThreadId,
            user_id: userId,
            role: "user",
            content: message,
            file_urls: fileUrls,
          })
          .select()
          .single();
        if (userMessageError) {
          console.error(`[MedAlpaca API] User message error:`, userMessageError);
          throw userMessageError;
        }
        // Save assistant message
        const storedPayload = structuredResponse && typeof structuredResponse === "object"
          ? {
              type: STRUCTURED_PAYLOAD_TYPE,
              version: STRUCTURED_PAYLOAD_VERSION,
              structuredResponse,
              text: responseText ?? undefined,
              modelKey,
              directAnswer: directAnswer ?? undefined,
              modelInfo: modelInfo ?? undefined,
              createdAt: new Date().toISOString(),
            } as Record<string, unknown>
          : null;
        const serializedContent = storedPayload
          ? JSON.stringify(storedPayload)
          : responseText ?? JSON.stringify(structuredResponse);
        const { data: assistantMessage, error: assistantMessageError } = await supabase
          .from("messages")
          .insert({
            thread_id: currentThreadId,
            user_id: userId,
            role: "assistant",
            content: serializedContent,
          })
          .select()
          .single();
        if (assistantMessageError) {
          console.error(`[MedAlpaca API] Assistant message error:`, assistantMessageError);
          throw assistantMessageError;
        }
        // Update usage tracking
        await supabase.rpc("update_usage_tracking", {
          p_user_id: userId,
          p_message_count: 1,
          p_file_upload_count: fileUrls?.length || 0,
        });
        return NextResponse.json({
          structuredResponse,
          directAnswer,
          modelInfo,
          response: responseText ?? undefined,
          threadId: currentThreadId,
          messageId: assistantMessage.id,
        });
      } catch (err) {
        console.error(`[MedAlpaca API] Unhandled error:`, err);
        return NextResponse.json({ error: `Backend unavailable: ${typeof err === 'string' ? err : JSON.stringify(err)}` }, { status: 502 });
      }
      return;
    }

    // --- Gemini fallback (legacy) ---
    try {
      // Map 'gemini' from frontend to 'gemini-2.5-flash' for Gemini API
      let geminiModel = "gemini-2.5-flash";
      if (modelKey && modelKey.startsWith("gemini")) {
        geminiModel = modelKey === "gemini" ? "gemini-2.5-flash" : modelKey;
      }
      const model = genAI.getGenerativeModel({ model: geminiModel });
      const fullPrompt = `${contextualPrompt}\n\nUser Message: ${message}${fileContent}`;
      const result = await model.generateContent(fullPrompt);
      const aiResponse = result.response.text();
      // Create or get thread
      let currentThreadId = threadId;
      if (!currentThreadId) {
        const { data: newThread, error: threadError } = await supabase
          .from("threads")
          .insert({
            user_id: userId,
            title: message.slice(0, 50) + (message.length > 50 ? "..." : ""),
          })
          .select()
          .single();
        if (threadError) throw threadError;
        currentThreadId = newThread.id;
      }
      // Save user message
      const { error: userMessageError } = await supabase
        .from("messages")
        .insert({
          thread_id: currentThreadId,
          user_id: userId,
          role: "user",
          content: message,
          file_urls: fileUrls,
        })
        .select()
        .single();
      if (userMessageError) throw userMessageError;
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
        .single();
      if (assistantMessageError) throw assistantMessageError;
      // Update usage tracking
      await supabase.rpc("update_usage_tracking", {
        p_user_id: userId,
        p_message_count: 1,
        p_file_upload_count: fileUrls?.length || 0,
      });
      return NextResponse.json({
        response: aiResponse,
        threadId: currentThreadId,
        messageId: assistantMessage.id,
      });
    } catch (error) {
      console.error("[Gemini API] Chat API error:", error);
      return NextResponse.json({ error: "Gemini API error or model unavailable" }, { status: 502 });
    }
  } catch (error) {
    console.error("[API] Unexpected error:", error);
  return NextResponse.json({ error: "Internal server error" }, { status: 500 });
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
