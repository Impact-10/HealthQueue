export const metadata = { title: 'General Health Dos & Don\'ts' }

interface Section {
  title: string
  dos: string[]
  donts: string[]
}

const sections: Section[] = [
  {
    title: 'Foundations',
    dos: [
      'Aim for consistent sleep (7–9 hrs)',
      'Build balanced plates: protein + fiber + healthy fats',
      'Hydrate across the day, not just at night',
      'Move daily: mix strength, cardio, mobility'
    ],
    donts: [
      'Skip meals then binge later',
      'Rely only on willpower—design your environment',
      'Chase extreme short-term “transformations”',
      'Ignore persistent fatigue or pain'
    ]
  },
  {
    title: 'Nutrition',
    dos: [
      'Prioritize minimally processed foods',
      'Include colorful vegetables & fruit most meals',
      'Plan protein across the day (not just dinner)',
      'Read labels: watch added sugars & trans fats'
    ],
    donts: [
      'Demonize entire food groups without medical need',
      'Use supplements as a substitute for real food',
      'Drink sugar-sweetened beverages daily',
      'Confuse marketing terms (natural, detox) with evidence'
    ]
  },
  {
    title: 'Lifestyle',
    dos: [
      'Schedule preventive checkups & screenings',
      'Take breaks from prolonged sitting',
      'Practice stress regulation (breath, walk, journal)',
      'Maintain social connection'
    ],
    donts: [
      'Self-diagnose serious symptoms via random forums',
      'Neglect mental health concerns',
      'Use alcohol to aid sleep—it fragments recovery',
      'Wait until burnout to rest'
    ]
  },
  {
    title: 'Medication & Safety',
    dos: [
      'Follow prescribed doses & timing',
      'Store medicines away from heat / kids',
      'Clarify interactions (e.g. supplements, grapefruit)',
      'Report adverse effects early'
    ],
    donts: [
      'Share prescription meds',
      'Mix multiple OTC drugs with same ingredient (e.g. acetaminophen)',
      'Stop antibiotics early when you feel “better”',
      'Start new supplements blindly'
    ]
  }
]

import { LandingHeader } from "@/components/layout/landing-header"

export default function DosDontsPage() {
  return (
    <div className="relative min-h-screen bg-gradient-to-b from-slate-950 via-slate-930 to-slate-900 text-slate-200">
      <LandingHeader />
      <main className="max-w-6xl mx-auto px-6 py-16 space-y-14">
        <section className="space-y-5">
          <div className="inline-flex items-center gap-2 rounded-full bg-teal-500/10 ring-1 ring-inset ring-teal-400/30 px-4 py-1 text-[11px] tracking-wide uppercase font-medium text-teal-300">Core Habits</div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">General Health Dos & Don&apos;ts</h1>
          <p className="text-sm text-slate-400 max-w-2xl leading-relaxed">Practical, evidence-aligned guidelines. Not exhaustive—individual needs vary; consult qualified professionals for personalized care.</p>
        </section>
        <div className="space-y-12">
          {sections.map((sec) => (
            <div key={sec.title} className="space-y-6">
              <h2 className="text-2xl font-semibold tracking-tight text-slate-100">{sec.title}</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div className="rounded-lg border border-emerald-500/25 bg-gradient-to-br from-emerald-500/10 via-emerald-500/5 to-emerald-500/5 p-6 backdrop-blur">
                  <h3 className="text-sm font-semibold mb-4 text-emerald-300 tracking-wide uppercase">Do</h3>
                  <ul className="space-y-2.5 text-[13px] leading-relaxed">
                    {sec.dos.map((d,i) => <li key={i} className="flex gap-2"><span className="text-emerald-400">•</span><span>{d}</span></li>)}
                  </ul>
                </div>
                <div className="rounded-lg border border-rose-500/25 bg-gradient-to-br from-rose-500/10 via-rose-500/5 to-rose-500/5 p-6 backdrop-blur">
                  <h3 className="text-sm font-semibold mb-4 text-rose-300 tracking-wide uppercase">Don&apos;t</h3>
                  <ul className="space-y-2.5 text-[13px] leading-relaxed">
                    {sec.donts.map((d,i) => <li key={i} className="flex gap-2"><span className="text-rose-400">•</span><span>{d}</span></li>)}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-slate-500">Disclaimer: Educational only; not medical advice. Seek licensed medical evaluation for personal conditions.</p>
      </main>
    </div>
  )
}
