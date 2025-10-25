import { NextResponse } from "next/server"

// Simple Overpass API proxy to avoid CORS issues from client
// Query clinics, hospitals, doctors, pharmacy within radius (meters) around lat/lon
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const lat = parseFloat(searchParams.get("lat") || "")
    const lon = parseFloat(searchParams.get("lon") || "")
    const radius = parseInt(searchParams.get("radius") || "3000", 10) // default 3km

    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      return NextResponse.json({ error: "lat and lon required" }, { status: 400 })
    }

  const amenities = ["clinic", "hospital"]
    // Overpass QL
    const query = `
      [out:json][timeout:25];
      (
        ${amenities
          .map((a) => `node["amenity"="${a}"](around:${radius},${lat},${lon}); way["amenity"="${a}"](around:${radius},${lat},${lon}); relation["amenity"="${a}"](around:${radius},${lat},${lon});`)
          .join("\n")}
      );
      out center tags;
    `

    const overpassUrl = "https://overpass-api.de/api/interpreter"
    const res = await fetch(overpassUrl, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ data: query }),
      // Overpass discourages aggressive caching; we can skip cache
      cache: "no-store",
    })

    if (!res.ok) {
      return NextResponse.json({ error: "Overpass error" }, { status: 502 })
    }

    const data = await res.json()

    const features = (data.elements || []).map((el: any) => {
      const latEl = el.lat || el.center?.lat
      const lonEl = el.lon || el.center?.lon
      return {
        id: el.id,
        type: el.tags?.amenity || "unknown",
        name: el.tags?.name || "Unnamed",
        lat: latEl,
        lon: lonEl,
        address: [el.tags?.addr_street, el.tags?.addr_housenumber, el.tags?.addr_city].filter(Boolean).join(" "),
        raw: el.tags || {}, // includes opening_hours if present
      }
    }).filter((f: any) => f.lat && f.lon)

    return NextResponse.json({ facilities: features })
  } catch (e) {
    return NextResponse.json({ error: "Server error" }, { status: 500 })
  }
}
