"use client"
import { useEffect, useState, useCallback, useRef } from "react"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"
import { Loader2, PlusCircle, RefreshCw, Send, Mic, MicOff, Stethoscope, MessageSquarePlus } from "lucide-react"
import { useSpeechInput } from "@/hooks/use-speech-input"
import { useTextToSpeech } from "@/hooks/use-text-to-speech"

interface DoctorThread { id: string; title: string; created_at: string; doctor_id: string | null }
interface DoctorMessage { id: string; role: "user" | "doctor" | "system"; content: string; created_at: string }
interface ThreadSummary { id: string; title: string; updated_at: string }
interface Doctor { id: string; full_name: string; specialty: string | null }

export default function DoctorChatClient({ userId }: { userId: string }) {
  const supabase = createClient()
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("consultations")
  const [doctors, setDoctors] = useState<Doctor[]>([])
  const [aiThreads, setAiThreads] = useState<ThreadSummary[]>([])
  const [doctorThreads, setDoctorThreads] = useState<DoctorThread[]>([])
  const [activeDoctorThread, setActiveDoctorThread] = useState<string | null>(null)
  const [messages, setMessages] = useState<DoctorMessage[]>([])
  const [newMessage, setNewMessage] = useState("")
  const [creating, setCreating] = useState(false)
  const [creatingSourceThread, setCreatingSourceThread] = useState<string | null>(null)
  const [selectedDoctor, setSelectedDoctor] = useState<string | null>(null)
  const [selectedSourceThread, setSelectedSourceThread] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const { supported: speechSupported, listening, interim, final, start: startSpeech, stop: stopSpeech, reset: resetSpeech } = useSpeechInput({ interim: true })
  const { supported: ttsSupported, speak } = useTextToSpeech()
  const speakNextRef = useRef(false)

  useEffect(() => {
    if (!listening) return
    setNewMessage(() => {
      const assembled = [final, interim].filter(Boolean).join(' ').replace(/\s+/g,' ').trim()
      return assembled
    })
  }, [final, interim, listening])

  const toggleMic = () => {
    if (!speechSupported) return
    if (listening) {
      stopSpeech()
    } else {
      resetSpeech(); startSpeech();
      speakNextRef.current = true
    }
  }

  const loadInitial = useCallback(async () => {
    setLoading(true)
    try {
      const [{ data: docs }, { data: threads }, { data: dthreads }] = await Promise.all([
        supabase.from("doctors").select("id, full_name, specialty").limit(50),
        supabase.from("threads").select("id, title, updated_at").order("updated_at", { ascending: false }).limit(30),
        supabase.from("doctor_threads").select("id, title, created_at, doctor_id").order("created_at", { ascending: false }).limit(30)
      ])
      setDoctors(docs || [])
      setAiThreads(threads || [])
      setDoctorThreads(dthreads || [])
    } finally {
      setLoading(false)
    }
  }, [supabase])

  const loadMessages = useCallback(async (doctorThreadId: string) => {
    const { data } = await supabase
      .from("doctor_messages")
      .select("id, role, content, created_at")
      .eq("doctor_thread_id", doctorThreadId)
      .order("created_at", { ascending: true })
    setMessages(data || [])
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [supabase])

  useEffect(() => { loadInitial() }, [loadInitial])
  useEffect(() => { if (activeDoctorThread) loadMessages(activeDoctorThread) }, [activeDoctorThread, loadMessages])

  const createDoctorThread = async () => {
    if (!selectedSourceThread) return
    setCreating(true)
    setCreatingSourceThread(selectedSourceThread)
    try {
      const res = await fetch("/api/doctor-thread", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sourceThreadId: selectedSourceThread, userId, doctorId: selectedDoctor }) })
      if (res.ok) {
        const data = await res.json()
        await loadInitial()
        setActiveDoctorThread(data.doctorThreadId)
        setSelectedSourceThread(null)
      }
    } finally {
      setCreating(false)
      setCreatingSourceThread(null)
    }
  }

  const sendMessage = async () => {
    if (!newMessage.trim() || !activeDoctorThread) return
    const temp: DoctorMessage = { id: `temp-${Date.now()}`, role: "user", content: newMessage, created_at: new Date().toISOString() }
    setMessages(prev => [...prev, temp])
    const content = newMessage
    setNewMessage("")
    const res = await fetch("/api/doctor-message", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ doctorThreadId: activeDoctorThread, content, userId }) })
    if (!res.ok) {
      setMessages(prev => prev.filter(m => m.id !== temp.id))
    } else {
      const { message, doctorReply } = await res.json()
      setMessages(prev => prev.map(m => m.id === temp.id ? message : m).concat(doctorReply ? [doctorReply] : []))
      if (ttsSupported && speakNextRef.current && doctorReply && doctorReply.role === 'doctor') {
        speakNextRef.current = false
        const toSpeak = doctorReply.content.replace(/\s+/g,' ').trim()
        if (toSpeak) speak(toSpeak)
      } else {
        speakNextRef.current = false
      }
    }
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const showOnboarding = !loading && doctorThreads.length === 0 && aiThreads.length === 0

  // Mock fallback doctors (UI prototype) if none from DB
  const mergedDoctors: Doctor[] = doctors.length > 0 ? doctors : [
    { id: "m1", full_name: "Dr. Alice Hart", specialty: "Cardiology" },
    { id: "m2", full_name: "Dr. Ben Lee", specialty: "Endocrinology" },
    { id: "m3", full_name: "Dr. Carol Singh", specialty: "General Medicine" },
    { id: "m4", full_name: "Dr. David Ortiz", specialty: "Sports Medicine" },
  ]

  function pickDoctor(id: string) {
    setSelectedDoctor(id)
    setActiveTab("consultations")
  }

  return (
  <div className="flex flex-col gap-5 min-h-[80vh] max-w-7xl mx-auto px-3 sm:px-5 pb-10">
      <div className="flex items-start gap-4 pt-4">
        <div className="flex items-center justify-center h-12 w-12 rounded-xl bg-gradient-to-br from-emerald-600/25 to-emerald-500/10 border border-emerald-700/40">
          <Stethoscope className="h-6 w-6 text-emerald-400" />
        </div>
        <div className="space-y-2">
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-slate-100">Doctor Consultation Workspace</h1>
          <p className="text-sm leading-relaxed text-slate-400 max-w-2xl">Upgrade an earlier AI conversation into a prepped consultation. We'll summarize key details so a doctor can review faster. (Prototype – real doctor replies not live yet.)</p>
        </div>
      </div>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-col flex-1">
        <TabsList className="w-fit bg-slate-900/60 border border-slate-700/60">
          <TabsTrigger value="consultations" className="data-[state=active]:bg-slate-800/80 data-[state=active]:text-white">Consultations</TabsTrigger>
          <TabsTrigger value="doctors" className="data-[state=active]:bg-slate-800/80 data-[state=active]:text-white">Doctors</TabsTrigger>
        </TabsList>
        <TabsContent value="consultations" className="flex-1 mt-4 flex flex-col gap-6">
          {showOnboarding && (
            <Card className="border border-slate-700/60 bg-slate-900/60 backdrop-blur">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-200 flex items-center gap-2"><MessageSquarePlus className="h-4 w-4 text-emerald-400" /> Create Your First Consultation</CardTitle>
              </CardHeader>
              <CardContent className="text-xs space-y-3 text-slate-400">
                <p>1. Go to the main chat and ask your health question as usual.</p>
                <p>2. Come back here, pick that earlier chat, and optionally choose a doctor.</p>
                <p>3. We prep a concise summary to help a doctor review faster.</p>
                <p className="text-[10px] text-slate-500">Real doctor messaging is coming soon.</p>
              </CardContent>
            </Card>
          )}
          <div className="flex flex-1 gap-6 overflow-hidden">
            <div className="w-80 flex flex-col gap-4 shrink-0">
              <Card className="flex flex-col h-[340px] border-slate-700/60 bg-slate-900/60 backdrop-blur">
                <CardHeader className="py-3"><CardTitle className="text-[13px] font-medium text-slate-200">Your Consultations</CardTitle></CardHeader>
                <CardContent className="pt-0 overflow-y-auto space-y-1 scrollbar-thin">
                  {doctorThreads.map(dt => (
                    <button key={dt.id} onClick={() => setActiveDoctorThread(dt.id)} className={cn("w-full text-left rounded-md px-3 py-2 text-[12px] border transition font-medium", dt.id === activeDoctorThread ? "bg-emerald-600/15 border-emerald-500/60 text-emerald-200" : "border-slate-700/60 hover:bg-slate-800/70 text-slate-300")}>{dt.title}</button>
                  ))}
                  {doctorThreads.length === 0 && <p className="text-xs text-slate-500 px-2 py-4">No consultations yet.</p>}
                </CardContent>
              </Card>
              <Card className="flex flex-col border-slate-700/60 bg-slate-900/60 backdrop-blur">
                <CardHeader className="py-3"><CardTitle className="text-[13px] font-medium text-slate-200">New Consultation</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-1">
                    <label className="text-[10px] uppercase tracking-wide font-medium text-slate-500">Doctor (Optional)</label>
                    <Select onValueChange={v => setSelectedDoctor(v)} value={selectedDoctor || undefined}>
                      <SelectTrigger className="h-8 text-[11px] bg-slate-800/70 border-slate-700/60"><SelectValue placeholder="No preference" /></SelectTrigger>
                      <SelectContent>
                        {mergedDoctors.map(d => <SelectItem key={d.id} value={d.id}>{d.full_name}{d.specialty ? ` — ${d.specialty}` : ""}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] uppercase tracking-wide font-medium text-slate-500">Previous Chat</label>
                    <Select onValueChange={v => setSelectedSourceThread(v)} value={selectedSourceThread || undefined}>
                      <SelectTrigger className="h-8 text-[11px] bg-slate-800/70 border-slate-700/60"><SelectValue placeholder="Pick a chat" /></SelectTrigger>
                      <SelectContent>
                        {aiThreads.length === 0 && <div className="px-2 py-1 text-[10px] text-slate-500">No chats yet</div>}
                        {aiThreads.map(t => <SelectItem key={t.id} value={t.id}>{t.title}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <Button size="sm" className="w-full bg-emerald-600 hover:bg-emerald-500" disabled={!selectedSourceThread || creating} onClick={createDoctorThread}>
                    {creating && creatingSourceThread === selectedSourceThread ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlusCircle className="h-4 w-4" />}<span className="ml-1">Start Consultation</span>
                  </Button>
                  <Button variant="outline" size="sm" className="w-full border-slate-600/60" onClick={loadInitial}><RefreshCw className="h-4 w-4 mr-1" />Refresh</Button>
                  <p className="text-[10px] leading-relaxed text-slate-500">We'll auto-summarize that chat for clinical context.</p>
                </CardContent>
              </Card>
            </div>
            <div className="flex-1 flex flex-col rounded-xl border border-slate-700/60 bg-gradient-to-b from-slate-900/70 via-slate-900/50 to-slate-900/40 backdrop-blur shadow-sm">
              {activeDoctorThread ? (
                <div className="flex flex-col h-full">
                  <div className="flex-1 overflow-y-auto p-6 space-y-5 scrollbar-thin">
                    {messages.map(m => (
                      <div key={m.id} className={cn("group relative rounded-lg p-4 pr-5 text-sm max-w-[720px] border shadow-sm transition", m.role === "user" ? "bg-slate-800/70 border-slate-700/60 ml-auto" : m.role === "doctor" ? "bg-emerald-700/25 border-emerald-600/50" : "bg-slate-800/50 border-slate-700/50")}> 
                        <div className="flex items-start gap-3">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1 text-slate-500">
                              <span className="text-[10px] uppercase tracking-wide font-medium">{m.role}</span>
                              <span className="text-[10px] text-slate-600/60">{/* timestamp placeholder */}</span>
                            </div>
                            <div className="whitespace-pre-wrap leading-relaxed text-slate-200">
                              {m.content}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    <div ref={bottomRef} />
                  </div>
                  <div className="border-t border-slate-700/60 bg-slate-900/60 p-4 flex flex-col gap-3">
                    <div className="flex gap-3">
                      <div className="flex-1 relative">
                        <Textarea value={newMessage} onChange={e => setNewMessage(e.target.value)} onKeyDown={e => { if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); }}} placeholder="Ask a follow‑up... (Shift+Enter = new line)" className="min-h-[70px] max-h-48 resize-none pr-20 bg-slate-800/70 border-slate-700/60 text-slate-200 placeholder:text-slate-500" />
                        {speechSupported && (
                          <Button type="button" variant={listening ? "default" : "ghost"} size="sm" aria-pressed={listening} className={"absolute top-2 right-12 h-8 w-8 p-0 " + (listening ? "bg-rose-600 hover:bg-rose-500 text-white animate-pulse" : "text-muted-foreground hover:text-foreground")} onClick={toggleMic} title={listening ? "Stop voice input" : "Start voice input"}>
                            {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                          </Button>
                        )}
                      </div>
                      <Button onClick={sendMessage} disabled={!newMessage.trim()} className="h-11 px-5 self-end bg-emerald-600 hover:bg-emerald-500"><Send className="h-4 w-4" /></Button>
                    </div>
                    <p className="text-[10px] text-slate-500 px-1">Prototype only – real doctor messaging coming soon.</p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                  <div className="text-center space-y-2">
                    <p className="text-slate-300 font-medium">No consultation selected</p>
                    <p className="text-xs max-w-xs text-slate-500">Start one from a previous chat using the panel on the left.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </TabsContent>
        <TabsContent value="doctors" className="mt-4 flex-1">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {mergedDoctors.map((d, i) => {
              const initials = d.full_name.split(/\s+/).map(p => p[0]).slice(0,2).join("")
              const rating = (4 + (i % 2 ? 0.5 : 0)).toFixed(1)
              return (
                <Card key={d.id} className="group relative overflow-hidden border-slate-700/60 bg-slate-900/60 backdrop-blur hover:border-emerald-500/60 transition">
                  <CardHeader className="pb-2 flex flex-row items-start gap-3">
                    <Avatar className="h-11 w-11 border">
                      <AvatarImage src={`https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(initials)}`} alt={d.full_name} />
                      <AvatarFallback>{initials}</AvatarFallback>
                    </Avatar>
                    <div className="space-y-1">
                      <CardTitle className="text-sm font-medium leading-tight text-slate-200">{d.full_name}</CardTitle>
                      <p className="text-[11px] text-slate-500">{d.specialty || "General Practice"}</p>
                      <p className="text-[10px] text-amber-400 font-medium">★ {rating}</p>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0 text-xs text-slate-400 space-y-3">
                    <p className="leading-relaxed">Evidence-based preventive care & patient-centered guidance.</p>
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" className="h-7 px-2 text-[11px] border-slate-600/60" onClick={() => pickDoctor(d.id)}>Select</Button>
                      <Button size="sm" className="h-7 px-2 text-[11px] bg-emerald-600 hover:bg-emerald-500" disabled>Profile</Button>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export type __DoctorChatClientProps = { userId: string }
