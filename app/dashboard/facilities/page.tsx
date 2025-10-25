import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { NearbyFacilities } from "@/components/facilities/nearby-facilities"

export const dynamic = "force-dynamic"
export const metadata = { title: 'Facilities' }

export default async function FacilitiesPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth/login")
  }

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto space-y-10">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Nearby Clinics & Hospitals</h1>
        <p className="text-sm text-muted-foreground max-w-3xl">Discover nearby medical facilities using OpenStreetMap / Overpass data. Toggle between list and map views, adjust your search radius, and open directions instantly.</p>
      </div>
      <section className="space-y-8">
        <NearbyFacilities />
        <div className="rounded-xl border bg-gradient-to-b from-background/60 to-background/30 backdrop-blur p-8 shadow-sm">
          <h2 className="text-xl font-medium mb-2">Need a structured consultation?</h2>
          <p className="text-sm text-muted-foreground mb-5 max-w-2xl">Convert any AI assistant chat into a doctor consultation with an auto-generated clinical-style summary that organizes your health context and recent discussion.</p>
          <a href="/dashboard/doctor-chat" className="inline-flex items-center gap-2 text-sm px-5 py-2.5 rounded-md bg-primary hover:bg-primary/90 text-primary-foreground font-medium shadow-sm transition">Go to Doctor Consultations →</a>
        </div>
      </section>
    </div>
  )
}
