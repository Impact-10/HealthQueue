import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import Link from "next/link"
export const dynamic = 'force-dynamic'
export const metadata = { title: 'Doctor Consultation' }

// @ts-ignore - recent file addition sometimes not picked up by incremental analyzer
import DoctorChatClient from "./doctor-chat-client"

export default async function DoctorChatPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/auth/login')
  return (
    <div className="flex flex-col h-full min-h-[80vh]">
      <div className="sticky top-0 z-20 border-b border-blue-900/70 bg-gradient-to-r from-[#0B1729] via-[#0F2038] to-[#132A47] backdrop-blur supports-[backdrop-filter]:bg-[#0F2038]/90">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/dashboard" className="text-sm font-medium text-slate-200 hover:text-white transition">← Back to Dashboard</Link>
            <span className="text-sm text-slate-400 hidden sm:inline">Doctor Consultation</span>
          </div>
          <div className="text-[11px] text-slate-400">
            Prototype — simulated responses
          </div>
        </div>
      </div>
      <div className="flex-1 bg-gradient-to-b from-slate-950 via-slate-950/95 to-slate-900/90">
        <DoctorChatClient userId={user.id} />
      </div>
    </div>
  )
}

