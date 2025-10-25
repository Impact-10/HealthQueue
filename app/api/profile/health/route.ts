import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

export async function GET(req: NextRequest) {
  try {
    const supabase = await createClient()
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    const { data, error } = await supabase
      .from('patient_profiles')
      .select('age, gender, medications, conditions, allergies, height_cm, weight_kg, bmi')
      .eq('user_id', user.id)
      .single()
    if (error && error.code !== 'PGRST116') throw error
    let profile = data || null
    if (profile && (profile as any).bmi == null && profile.height_cm && profile.weight_kg) {
      const h = Number(profile.height_cm)
      const w = Number(profile.weight_kg)
      if (h > 0) {
        ;(profile as any).bmi = Number((w / Math.pow(h / 100, 2)).toFixed(1))
      }
    }
    return NextResponse.json({ profile })
  } catch (e: any) {
    return NextResponse.json({ error: 'Failed to fetch profile' }, { status: 500 })
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { age, gender, medications, conditions, allergies, height_cm, weight_kg } = body || {}
    const supabase = await createClient()
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

    // Upsert pattern
    const { data, error } = await supabase
      .from('patient_profiles')
      .upsert({
        user_id: user.id,
        age: age ?? null,
        gender: gender ?? null,
        medications: medications ?? null,
        conditions: conditions ?? null,
        allergies: allergies ?? null,
        height_cm: height_cm ?? null,
        weight_kg: weight_kg ?? null,
      }, { onConflict: 'user_id' })
      .select('age, gender, medications, conditions, allergies, height_cm, weight_kg, bmi')
      .single()
    if (error) throw error
    let profile = data
    if (profile && (profile as any).bmi == null && profile.height_cm && profile.weight_kg) {
      const h = Number(profile.height_cm)
      const w = Number(profile.weight_kg)
      if (h > 0) {
        ;(profile as any).bmi = Number((w / Math.pow(h / 100, 2)).toFixed(1))
      }
    }
    return NextResponse.json({ profile })
  } catch (e: any) {
    return NextResponse.json({ error: 'Failed to upsert profile' }, { status: 500 })
  }
}