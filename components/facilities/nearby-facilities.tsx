"use client"

import { useEffect, useState, useMemo, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Loader2, Map, List, RefreshCw, Navigation } from "lucide-react"
import dynamic from "next/dynamic"

interface Facility {
  id: number | string
  type: string
  name: string
  lat: number
  lon: number
  address?: string
  raw: Record<string, any>
}

type ViewMode = "list" | "map"

// Dynamically import react-leaflet map components to avoid SSR issues
const MapContainer = dynamic(() => import("./facility-map").then(m => m.FacilityMap), { ssr: false, loading: () => <div className="h-96 flex items-center justify-center text-slate-400">Loading map…</div> })

export function NearbyFacilities({ active = true }: { active?: boolean }) {
  const [coords, setCoords] = useState<{ lat: number; lon: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [facilities, setFacilities] = useState<Facility[]>([])
  const [loading, setLoading] = useState(false)
  const [radius, setRadius] = useState(3000)
  const [view, setView] = useState<ViewMode>("list")
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!coords) return
    fetchFacilities()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coords, radius])

  useEffect(() => {
    if (!active) return
    if (!navigator.geolocation) {
      setError("Geolocation not supported in this browser.")
      return
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => setCoords({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      (err) => {
        setError("Unable to retrieve your location. Allow location access and retry.")
        console.error(err)
      },
      { enableHighAccuracy: true, timeout: 10000 }
    )
  }, [active])

  const fetchFacilities = async () => {
    if (!coords) return
    setLoading(true)
    setError(null)
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const params = new URLSearchParams({ lat: String(coords.lat), lon: String(coords.lon), radius: String(radius) })
      const res = await fetch(`/api/places?${params.toString()}`, { signal: controller.signal })
      if (!res.ok) throw new Error("Failed to fetch facilities")
      const data = await res.json()
      setFacilities(data.facilities || [])
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const distance = (lat1: number, lon1: number, lat2: number, lon2: number) => {
    const R = 6371e3
    const toRad = (d: number) => (d * Math.PI) / 180
    const dLat = toRad(lat2 - lat1)
    const dLon = toRad(lon2 - lon1)
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
    return R * c
  }

  // Very lightweight opening_hours parser (supports basic patterns like "Mo-Su 08:00-20:00" or "24/7")
  const isOpenNow = (oh?: string): boolean | null => {
    if (!oh) return null
    const now = new Date()
    const weekday = ["Su","Mo","Tu","We","Th","Fr","Sa"][now.getDay()]
    const minutesNow = now.getHours() * 60 + now.getMinutes()
    if (oh.trim() === "24/7") return true
    // Split on ; for multiple rules and evaluate first matching simple rule
    const rules = oh.split(";").map(r => r.trim()).filter(Boolean)
    for (const rule of rules) {
      // e.g., "Mo-Fr 08:00-18:00", "Sa-Su 10:00-16:00", "Su 09:00-12:00"
      const match = rule.match(/^([A-Za-z,-]+)\s+(\d{1,2}:\d{2})-(\d{1,2}:\d{2})$/)
      if (match) {
        const days = match[1]
        const hours = match[2]
        const end = match[3]
        // Expand days range
        const daySet = new Set<string>()
        days.split(",").forEach((part: string) => {
          const range = part.split("-")
          const order = ["Mo","Tu","We","Th","Fr","Sa","Su"]
          if (range.length === 2) {
            const [start, stop] = range
            const si = order.indexOf(start)
            const ei = order.indexOf(stop)
            if (si !== -1 && ei !== -1) {
              if (si <= ei) {
                for (let i=si;i<=ei;i++) daySet.add(order[i])
              } else { // wrap (rare)
                for (let i=si;i<order.length;i++) daySet.add(order[i])
                for (let i=0;i<=ei;i++) daySet.add(order[i])
              }
            }
          } else {
            daySet.add(part)
          }
        })
        if (!daySet.has(weekday)) continue
        const [sh, sm] = hours.split(":").map(Number)
        const [eh, em] = end.split(":").map(Number)
        const startMin = sh * 60 + sm
        const endMin = eh * 60 + em
        if (minutesNow >= startMin && minutesNow <= endMin) return true
      }
    }
    return false
  }

  const enriched = useMemo(() => {
    if (!coords) return [] as (Facility & { distance: number; open: boolean | null })[]
    return facilities
      .map((f) => ({
        ...f,
        distance: distance(coords.lat, coords.lon, f.lat, f.lon),
        open: isOpenNow(f.raw?.opening_hours),
      }))
      .sort((a, b) => a.distance - b.distance)
  }, [facilities, coords])

  return (
    <div className="space-y-8">
  <div className="flex flex-wrap gap-3 items-end">
        <div>
          <label className="block text-xs uppercase tracking-wide text-slate-400 mb-1">Radius (m)</label>
          <Input
            type="number"
            min={500}
            max={10000}
            step={100}
            value={radius}
            onChange={(e) => setRadius(Number(e.target.value))}
            className="w-32 bg-input/40 border-border text-foreground placeholder:text-muted-foreground/60"
          />
        </div>
        <Button variant={view === "list" ? "default" : "outline"} onClick={() => setView("list")} className="min-w-20"> <List className="h-4 w-4 mr-1"/> List</Button>
        <Button variant={view === "map" ? "default" : "outline"} onClick={() => setView("map")} className="min-w-20"> <Map className="h-4 w-4 mr-1"/> Map</Button>
        <Button variant="outline" onClick={fetchFacilities} disabled={loading || !coords} className="gap-2">
          <RefreshCw className="h-4 w-4" /> {loading ? "Loading" : "Refresh"}
        </Button>
        {coords && (
          <div className="group relative">
            <div className="text-xs text-muted-foreground/70 px-2 py-1 rounded bg-muted/30 group-hover:bg-muted/50 flex items-center gap-1 transition-colors">
              <Navigation className="h-3 w-3" /> Location
            </div>
            <div className="absolute left-0 mt-1 hidden group-hover:flex bg-popover border border-border rounded p-2 text-[10px] text-muted-foreground shadow-sm">
              {coords.lat.toFixed(4)}, {coords.lon.toFixed(4)}
            </div>
          </div>
        )}
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}
      {loading && (
        <div className="flex items-center gap-2 text-slate-400 text-sm">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading facilities…
        </div>
      )}

      {!loading && !error && enriched.length === 0 && coords && (
        <p className="text-slate-400 text-sm">No facilities found within {radius}m.</p>
      )}

      {view === "list" && enriched.length > 0 && (
  <div className="grid gap-8 md:grid-cols-2">
          {enriched.map((f) => {
            const facilityName = f.name || 'Unnamed Facility'
            return (
              <Card
                key={f.id}
                className="relative group hover:border-primary/40 transition-colors flex flex-col border-border/60 bg-card/60 backdrop-blur supports-[backdrop-filter]:bg-card/40 min-h-[230px]"
              >
                <CardHeader className="pb-3 pt-5 px-5 flex-shrink-0">
                  <div className="flex items-start justify-between gap-3">
                    <CardTitle
                      title={facilityName}
                      className="text-[0.95rem] font-semibold text-card-foreground leading-snug line-clamp-2 break-words"
                    >
                      {facilityName}
                    </CardTitle>
                    <span
                      className={
                        'shrink-0 text-[11px] font-medium tracking-wide px-2.5 py-1 rounded-md ring-1 ring-inset capitalize ' +
                        (f.type.includes('clinic')
                          ? 'bg-emerald-500/12 text-emerald-400 ring-emerald-500/25'
                          : f.type.includes('hospital')
                          ? 'bg-rose-500/12 text-rose-400 ring-rose-500/25'
                          : 'bg-primary/12 text-primary ring-primary/25')
                      }
                    >
                      {f.type}
                    </span>
                  </div>
                </CardHeader>
                <CardContent className="text-[13px] text-muted-foreground flex flex-col gap-5 flex-1 px-5 pb-5">
                  <div className="space-y-3">
                    {f.address && (
                      <p className="text-muted-foreground/80 line-clamp-1 tracking-wide" title={f.address}>
                        {f.address}
                      </p>
                    )}
                    <div className="flex items-center gap-3 flex-wrap text-[12px]">
                      <p className="text-muted-foreground/70 font-medium">{(f.distance / 1000).toFixed(2)} km away</p>
                      {f.open !== null && (
                        <span
                          className={
                            'text-[11px] px-2.5 py-0.5 rounded-md font-medium tracking-wide ' +
                            (f.open
                              ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-inset ring-emerald-500/25'
                              : 'bg-rose-500/15 text-rose-400 ring-1 ring-inset ring-rose-500/25')
                          }
                        >
                          {f.open ? 'Open' : 'Closed'}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="mt-auto pt-1 flex flex-wrap gap-3">
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-8 text-[12px] px-3.5 shadow-xs hover:shadow-sm transition"
                      asChild
                    >
                      <a
                        href={`https://www.openstreetmap.org/?mlat=${f.lat}&mlon=${f.lon}#map=17/${f.lat}/${f.lon}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        OSM
                      </a>
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-8 text-[12px] px-3.5 shadow-xs hover:shadow-sm transition"
                      asChild
                    >
                      <a
                        href={`https://www.google.com/maps/search/?api=1&query=${f.lat},${f.lon}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Maps
                      </a>
                    </Button>
                    {coords && (
                      <Button
                        size="sm"
                        variant="default"
                        className="h-8 text-[12px] px-3.5 shadow-sm hover:shadow focus-visible:ring-ring/60 flex items-center gap-1"
                        asChild
                      >
                        <a
                          href={`https://www.google.com/maps/dir/?api=1&origin=${coords.lat},${coords.lon}&destination=${f.lat},${f.lon}&travelmode=driving`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <span className="inline-block">Directions</span>
                        </a>
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      {view === "map" && coords && (
        <div className="h-[500px] rounded-md overflow-hidden border border-slate-700">
          <MapContainer userCoords={coords} facilities={enriched} />
        </div>
      )}
    </div>
  )
}
