export const metadata = { title: 'First Aid Basics' }

import { LandingHeader } from "@/components/layout/landing-header"

interface AidSection { title: string; steps: string[]; caution?: string; whenToSeek?: string[] }

const sections: AidSection[] = [
  {
    title: 'Severe Bleeding',
    steps: [
      'Apply firm, direct pressure with clean cloth or bandage',
      'Do not remove soaked layers—add more on top',
      'Elevate injured area if no fracture suspected',
      'If life-threatening and trained: apply tourniquet 5–7 cm above wound'
    ],
    caution: 'Never use a tourniquet on joints; note the time applied',
    whenToSeek: ['Bleeding spurts or soaks through repeatedly', 'Signs of shock: pale, clammy, rapid pulse']
  },
  {
    title: 'Burns (Thermal)',
    steps: [
      'Cool burn under cool (not icy) running water 10–20 min',
      'Remove tight items (rings) before swelling',
      'Cover loosely with sterile non‑stick dressing',
      'Hydrate if large area affected'
    ],
    caution: 'Do not apply butter, oils, toothpaste, or ice',
    whenToSeek: ['Burn larger than palm', 'Face, hands, genitals, major joints', 'Electrical or chemical burn']
  },
  {
    title: 'Choking (Adult/Child Conscious)',
    steps: [
      'Ask: “Are you choking?” If cannot speak/cough: act',
      'Give 5 back blows (heel of hand between shoulder blades)',
      'Then 5 abdominal thrusts (Heimlich) above navel',
      'Alternate 5 & 5 until object expelled or unresponsive'
    ],
    caution: 'Pregnant or obese: use chest thrusts at sternum level',
    whenToSeek: ['Becomes unconscious (begin CPR)', 'Persistent breathing difficulty after removal']
  },
  {
    title: 'Possible Heart Attack',
    steps: [
      'Call emergency services immediately',
      'Keep person calm, seated, loosen tight clothing',
      'If not allergic & no contraindication: 160–325 mg chewable aspirin',
      'Be ready to start CPR if collapse occurs'
    ],
    caution: 'Do not give aspirin if bleeding risk, allergy, or recent GI bleed',
    whenToSeek: ['Chest pressure >5 min or radiating pain', 'Shortness of breath, sweating, nausea']
  },
  {
    title: 'Suspected Stroke (FAST)',
    steps: [
      'Face: ask to smile—droop?',
      'Arms: raise both—one drift?',
      'Speech: slurred or strange?',
      'Time: call emergency services immediately'
    ],
    caution: 'Note onset time—critical for treatment decisions'
  }
]

export default function FirstAidPage() {
  return (
    <div className="relative min-h-screen bg-gradient-to-b from-slate-950 via-slate-930 to-slate-900 text-slate-200">
      <LandingHeader />
      <main className="max-w-5xl mx-auto px-6 py-16 space-y-14">
        <section className="space-y-5">
          <div className="inline-flex items-center gap-2 rounded-full bg-red-500/10 ring-1 ring-inset ring-red-400/30 px-4 py-1 text-[11px] tracking-wide uppercase font-medium text-red-300">Emergency Primer</div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">First Aid Basics</h1>
          <p className="text-sm text-slate-400 max-w-2xl leading-relaxed">Immediate actions that can stabilize common emergencies before professional medical care arrives. Not a substitute for certified training.</p>
        </section>
        <div className="space-y-10">
          {sections.map(sec => (
            <div key={sec.title} className="rounded-lg border border-slate-700/70 bg-slate-800/40 backdrop-blur p-6 space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <h2 className="text-xl font-semibold text-slate-100 tracking-tight">{sec.title}</h2>
                {sec.caution && <span className="text-[11px] px-2.5 py-1 rounded-md bg-amber-500/15 text-amber-300 ring-1 ring-inset ring-amber-400/30">Caution</span>}
              </div>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wide">Steps</h3>
                  <ol className="space-y-2 text-[13px] list-decimal list-inside text-slate-400">
                    {sec.steps.map((s,i) => <li key={i}>{s}</li>)}
                  </ol>
                </div>
                <div className="space-y-4">
                  {sec.whenToSeek && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wide">Seek Emergency Help If</h3>
                      <ul className="space-y-2 text-[13px] text-slate-400 list-disc list-inside">
                        {sec.whenToSeek.map((w,i) => <li key={i}>{w}</li>)}
                      </ul>
                    </div>
                  )}
                  {sec.caution && <p className="text-[11px] text-amber-400/80 leading-relaxed">{sec.caution}</p>}
                </div>
              </div>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-slate-500">Disclaimer: Educational reference only. Obtain certified first aid training for comprehensive preparedness.</p>
      </main>
    </div>
  )
}
