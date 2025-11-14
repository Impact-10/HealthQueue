/* eslint-disable @typescript-eslint/no-explicit-any */
import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"

const BACKEND_URL = process.env.DIAGNOSAI_BACKEND_URL || "http://localhost:8000"

const STRUCTURED_PAYLOAD_TYPE = "structured-response"
const STRUCTURED_PAYLOAD_VERSION = 1

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

    // --- BERT QA model routing ---
    if (modelKey === "bert-qa") {
  // Ensure existingThread is always defined to avoid TS errors
  let existingThread: any = null;
      // Use BERT extractive QA with retrieval
      try {
        const backendResp = await fetch(`${BACKEND_URL}/api/qa`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            question: message,
            k: 10,  // Retrieve 10 passages
            alpha: 0.6,  // Higher semantic weight
            max_answer_length: 150,
          }),
        });
        if (!backendResp.ok) {
          const err = await backendResp.text();
          console.error(`[BERT QA API] Backend error:`, err);
          return NextResponse.json({ error: `Backend error: ${err}` }, { status: 502 });
        }
        
        const result = await backendResp.json();
        
        if (result.mode === "success" || result.mode === "low_confidence") {
          const responseText = result.answer || result.message || "Retrieved relevant passages";
          const structuredResponse = {
            type: STRUCTURED_PAYLOAD_TYPE,
            version: STRUCTURED_PAYLOAD_VERSION,
            mode: result.mode === "low_confidence" ? "low_confidence" : "extractive_qa",
            content: {
              answer: result.answer,
              raw_answer: result.raw_answer,
              confidence: result.confidence,
              top_passages: result.top_passages || undefined,
            },
            metadata: {
              model_name: "BERT QA",
              source: result.source,
              retrieval: result.retrieval,
              safety_warning: result.safety_warning || undefined,
            },
          };
          
          // Save to database if thread logic is present
          if (typeof existingThread !== "undefined" && existingThread) {
            await supabase.from("messages").insert({
              thread_id: existingThread.thread_id,
              role: "user",
              content: message,
              created_at: new Date().toISOString(),
            });
            
            await supabase.from("messages").insert({
              thread_id: existingThread.thread_id,
              role: "assistant",
              content: responseText,
              metadata: structuredResponse,
              created_at: new Date().toISOString(),
            });
            
            await supabase.from("threads").update({
              last_message: responseText,
              updated_at: new Date().toISOString(),
            }).eq("thread_id", existingThread.thread_id);
          }
          
          return NextResponse.json({
            response: responseText,
            structuredResponse,
            model: {
              key: "bert-qa",
              name: "BERT QA",
              mode: "extractive_qa",
              badge: "Medical QA",
            },
          });
        } else if (result.mode === "blocked") {
          // Query was blocked by safety filter
          return NextResponse.json({ 
            error: result.message || "Query blocked by safety filter",
            mode: result.mode,
            severity: result.severity,
          }, { status: 400 });
        } else {
          // Fallback if QA failed
          return NextResponse.json({ 
            error: result.message || "Could not extract answer",
            mode: result.mode,
          }, { status: 500 });
        }
      } catch (err) {
        console.error(`[BERT QA API] Unhandled error:`, err);
        return NextResponse.json({ error: "Internal BERT QA error" }, { status: 500 });
      }
    }
    
    // Only BERT QA is supported
    return NextResponse.json({ error: "Only BERT QA model is supported. Please use modelKey: 'bert-qa'" }, { status: 400 });
  } catch (error) {
    console.error("[API] Unexpected error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

