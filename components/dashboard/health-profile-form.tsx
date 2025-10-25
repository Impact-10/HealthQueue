"use client"

import { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'

// Helper to safely parse numeric inputs (accepts commas, spaces, trailing dot)
const parseNum = (value: unknown): number | undefined => {
  if (value === undefined || value === null) return undefined
  if (typeof value === 'number' && !Number.isNaN(value)) return value
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  if (trimmed === '') return undefined
  // Replace commas, allow trailing period removed
  const cleaned = trimmed.replace(/,/g, '').replace(/\.$/, '')
  if (cleaned === '') return undefined
  const n = Number(cleaned)
  return Number.isNaN(n) ? undefined : n
}

const schema = z.object({
  age: z.preprocess(parseNum, z.number({ invalid_type_error: 'Enter a valid age' }).int('Whole number only').min(0,'Too low').max(120,'Unrealistic').optional()),
  gender: z.string().optional(),
  height_cm: z.preprocess(parseNum, z.number({ invalid_type_error: 'Enter a valid height' }).min(40,'Too short').max(260,'Too tall').optional()),
  weight_kg: z.preprocess(parseNum, z.number({ invalid_type_error: 'Enter a valid weight' }).min(3,'Too low').max(400,'Too high').optional()),
  medications: z.string().optional(),
  conditions: z.string().optional(),
  allergies: z.string().optional(),
})

type FormValues = z.infer<typeof schema>

interface HealthProfileFormProps {
  initialProfile?: Partial<FormValues & { bmi?: number | null }>;
}

export function HealthProfileForm({ initialProfile }: HealthProfileFormProps) {
  const [loading, setLoading] = useState(false)
  const [initialLoad, setInitialLoad] = useState(true)
  const [message, setMessage] = useState<string|null>(null)
  const [error, setError] = useState<string|null>(null)
  const [bmi, setBmi] = useState<number | null>(null)
  const { register, handleSubmit, setValue, watch, reset, formState: { errors, isDirty } } = useForm<FormValues>({ resolver: zodResolver(schema), mode: 'onBlur' })

  const height = watch('height_cm')
  const weight = watch('weight_kg')

  useEffect(() => {
    if (height != null && weight != null && height !== undefined && weight !== undefined && height !== 0) {
      const calc = Number(weight) / Math.pow(Number(height)/100, 2)
      if (isFinite(calc)) {
        setBmi(Number(calc.toFixed(1)))
        return
      }
    }
    setBmi(null)
  }, [height, weight])

  useEffect(() => {
    let cancelled = false
    const hydrate = async () => {
      if (initialProfile) {
        reset(initialProfile as any)
        if (initialProfile.bmi) setBmi(initialProfile.bmi)
        setInitialLoad(false)
        return
      }
      try {
        const res = await fetch('/api/profile/health')
        if (!cancelled && res.ok) {
          const data = await res.json()
          if (data.profile) {
            reset(data.profile)
            if (data.profile.bmi) setBmi(data.profile.bmi)
          }
        }
      } finally {
        if (!cancelled) setInitialLoad(false)
      }
    }
    hydrate()
    return () => { cancelled = true }
  }, [reset, initialProfile])

  const onSubmit = async (values: FormValues) => {
    setLoading(true); setMessage(null); setError(null)
    try {
      const res = await fetch('/api/profile/health', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Save failed')
      reset(data.profile)
      if (data.profile?.bmi) setBmi(data.profile.bmi)
      setMessage('Profile saved')
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="bg-card/60 backdrop-blur border border-border shadow-sm">
      <CardHeader>
        <CardTitle className="text-card-foreground text-lg">Health Profile</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {error && <Alert className="border border-destructive/40 bg-destructive/10"><AlertDescription className="text-destructive-foreground/90 text-sm">{error}</AlertDescription></Alert>}
        {message && <Alert className="border border-emerald-500/40 bg-emerald-500/10"><AlertDescription className="text-emerald-300 text-sm">{message}</AlertDescription></Alert>}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">Age</label>
              <Input inputMode="numeric" placeholder="e.g. 34" {...register('age')} className="bg-input/40" disabled={loading || initialLoad} />
              {errors.age && <p className="text-xs text-destructive mt-1">{errors.age.message}</p>}
            </div>
            <div>
              <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">Gender</label>
              <Select onValueChange={v => setValue('gender', v, { shouldDirty: true })} value={(watch('gender') as string) || ''}>
                <SelectTrigger className="bg-input/40"><SelectValue placeholder="Select" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="male">Male</SelectItem>
                  <SelectItem value="female">Female</SelectItem>
                  <SelectItem value="non-binary">Non-binary</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                  <SelectItem value="prefer-not-to-say">Prefer not to say</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">Height (cm)</label>
              <Input inputMode="decimal" placeholder="e.g. 172" {...register('height_cm')} className="bg-input/40" disabled={loading || initialLoad} />
              {errors.height_cm && <p className="text-xs text-destructive mt-1">{errors.height_cm.message}</p>}
            </div>
            <div>
              <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">Weight (kg)</label>
              <Input inputMode="decimal" placeholder="e.g. 68.5" {...register('weight_kg')} className="bg-input/40" disabled={loading || initialLoad} />
              {errors.weight_kg && <p className="text-xs text-destructive mt-1">{errors.weight_kg.message}</p>}
            </div>
            <div className="sm:col-span-2 grid grid-cols-3 gap-4 items-end">
              <div className="col-span-1">
                <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">BMI</label>
                <div className="text-sm font-medium text-foreground/90 bg-input/30 border border-border rounded h-10 flex items-center px-3">{bmi ?? '—'}</div>
              </div>
              <div className="col-span-2 text-xs text-muted-foreground leading-relaxed">
                BMI is auto-calculated. This is a screening metric and not a diagnosis.
              </div>
            </div>
          </div>
          <div className="grid gap-4">
            <div>
              <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">Medications</label>
              <Textarea rows={2} placeholder="List current medications" {...register('medications')} className="bg-input/40 resize-y" />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">Conditions</label>
              <Textarea rows={2} placeholder="Chronic conditions or diagnoses" {...register('conditions')} className="bg-input/40 resize-y" />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1 uppercase tracking-wide text-muted-foreground">Allergies</label>
              <Textarea rows={2} placeholder="Allergies (e.g. penicillin, peanuts)" {...register('allergies')} className="bg-input/40 resize-y" />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Button type="submit" disabled={loading || initialLoad || !isDirty} className="shadow-sm">
              {loading ? 'Saving...' : 'Save Profile'}
            </Button>
            {isDirty && !loading && (
              <Button type="button" variant="ghost" onClick={() => { reset(); setMessage(null); setError(null) }} className="text-xs">Reset</Button>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
