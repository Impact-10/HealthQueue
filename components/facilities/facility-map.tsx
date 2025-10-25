"use client"

import { MapContainer as LeafletMap, TileLayer, Marker, Popup, CircleMarker } from "react-leaflet"
import "leaflet/dist/leaflet.css"
import L from "leaflet"
import { useMemo } from "react"

interface FacilityMapProps {
  userCoords: { lat: number; lon: number }
  facilities: Array<{ id: string | number; name: string; lat: number; lon: number; type: string; distance: number }>
}

// Generate a colored circular badge with a type letter for clinic/hospital/pharmacy/other
function typeLetter(t: string) {
  const lower = t.toLowerCase()
  if (lower.includes('hospital')) return 'H'
  if (lower.includes('clinic')) return 'C'
  if (lower.includes('pharm')) return 'P'
  return '•'
}

function typeColor(t: string) {
  const lower = t.toLowerCase()
  if (lower.includes('hospital')) return '#dc2626' // red
  if (lower.includes('clinic')) return '#059669'   // emerald
  if (lower.includes('pharm')) return '#d97706'    // amber
  return '#6366f1'                                 // indigo fallback
}

const facilityIcon = (t: string) => {
  const c = typeColor(t)
  const letter = typeLetter(t)
  return L.divIcon({
    className: "",
    html: `<div style="display:flex;align-items:center;justify-content:center;font-weight:600;font-size:10px;color:white;background:${c};width:20px;height:20px;border:2px solid #fff;box-shadow:0 0 0 1px ${c};border-radius:6px">${letter}</div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    popupAnchor: [0, -10]
  })
}

export function FacilityMap({ userCoords, facilities }: FacilityMapProps) {
  const center = useMemo(() => [userCoords.lat, userCoords.lon] as [number, number], [userCoords])

  return (
    <LeafletMap center={center} zoom={13} style={{ height: "100%", width: "100%" }} scrollWheelZoom={true}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <CircleMarker center={center} radius={8} pathOptions={{ color: "#3b82f6", fillColor: "#3b82f6", fillOpacity: 0.9 }} />
      {facilities.map((f) => (
        <Marker key={f.id} position={[f.lat, f.lon]} icon={facilityIcon(f.type)}> 
          <Popup>
            <div className="space-y-1">
              <h3 className="font-medium text-sm text-foreground/90 dark:text-foreground">{f.name}</h3>
              <p className="text-xs text-muted-foreground dark:text-muted-foreground capitalize">{f.type}</p>
              <p className="text-xs text-muted-foreground/80">{(f.distance / 1000).toFixed(2)} km away</p>
              <div className="flex gap-3 pt-1 flex-wrap items-center">
                <a
                  href={`https://www.openstreetmap.org/?mlat=${f.lat}&mlon=${f.lon}#map=17/${f.lat}/${f.lon}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-primary hover:underline"
                >
                  OSM
                </a>
                <a
                  href={`https://www.google.com/maps/search/?api=1&query=${f.lat},${f.lon}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-primary hover:underline"
                >
                  Maps
                </a>
                <a
                  href={`https://www.google.com/maps/dir/?api=1&origin=${userCoords.lat},${userCoords.lon}&destination=${f.lat},${f.lon}&travelmode=driving`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[11px] font-medium px-2.5 py-1 rounded-md bg-primary/15 text-primary ring-1 ring-inset ring-primary/30 hover:bg-primary/25 transition-colors"
                >
                  Directions
                </a>
              </div>
            </div>
          </Popup>
        </Marker>
      ))}
      {/* Legend overlay */}
      <div className="leaflet-top leaflet-right">
        <div className="m-2 rounded-lg bg-slate-900/80 backdrop-blur border border-slate-700 px-3 py-2 space-y-1 shadow text-[10px] text-slate-200">
          <p className="font-semibold text-[11px] tracking-wide">Legend</p>
          <div className="flex items-center gap-2"><span className="inline-flex items-center justify-center text-[9px] font-bold text-white bg-[#dc2626] w-4 h-4 rounded-[4px]">H</span><span>Hospital</span></div>
          <div className="flex items-center gap-2"><span className="inline-flex items-center justify-center text-[9px] font-bold text-white bg-[#059669] w-4 h-4 rounded-[4px]">C</span><span>Clinic</span></div>
          <div className="flex items-center gap-2"><span className="inline-flex items-center justify-center text-[9px] font-bold text-white bg-[#d97706] w-4 h-4 rounded-[4px]">P</span><span>Pharmacy</span></div>
          <div className="flex items-center gap-2"><span className="inline-flex items-center justify-center text-[9px] font-bold text-white bg-[#6366f1] w-4 h-4 rounded-[4px]">•</span><span>Other</span></div>
        </div>
      </div>
    </LeafletMap>
  )
}
