export const metadata = { title: 'Health Myths' }

const myths: { myth: string; fact: string }[] = [
  {
    myth: 'You need a detox juice cleanse to remove toxins',
    fact: 'Your liver, kidneys, lungs, skin and gut already detox continuously. A balanced diet, hydration, sleep and movement support them better than restrictive cleanses.'
  },
  {
    myth: 'Carbs are bad for you',
    fact: 'Whole-food carbohydrates (fruits, vegetables, legumes, whole grains) provide fiber, micronutrients and energy. Ultra‑processed refined carbs are the real issue.'
  },
  {
    myth: 'High-protein diets damage healthy kidneys',
    fact: 'In people with normal kidney function, higher protein intake is generally safe. Those with kidney disease need clinical guidance.'
  },
  {
    myth: 'You must drink 8 glasses of water a day',
    fact: 'Hydration needs vary with climate, diet and activity. Thirst, urine color (pale straw) and overall well‑being are better guides.'
  },
  {
    myth: 'Natural = Safe',
    fact: 'Plenty of natural substances are harmful (e.g. certain mushrooms). “Natural” is a marketing term, not a safety guarantee.'
  },
  {
    myth: 'Spot reduction (abs, thighs) works',
    fact: 'You cannot selectively burn fat in one area. Consistent nutrition, resistance training and overall energy balance drive body composition changes.'
  },
  {
    myth: 'Eggs dramatically raise heart risk',
    fact: 'For most people eggs in moderation are nutrient-dense. Overall dietary pattern matters more than a single food.'
  },
  {
    myth: 'Late night eating automatically causes weight gain',
    fact: 'Total intake, food quality and sleep disruption matter more than the clock. Grazing late can lead to excess calories—thats the real driver.'
  },
  {
    myth: 'Sweating more = better detox / more fat burned',
    fact: 'Sweat is for temperature regulation. Fluid loss on the scale is not fat loss.'
  },
  {
    myth: 'Vitamins give you energy',
    fact: 'Calories (macronutrients) provide energy. Micronutrients enable metabolic processes but dont directly “energize” you when taken in excess.'
  }
]

import { LandingHeader } from "@/components/layout/landing-header"

export default function HealthMythsPage() {
  return (
    <div className="relative min-h-screen bg-gradient-to-b from-slate-950 via-slate-930 to-slate-900 text-slate-200">
      <LandingHeader />
      <main className="max-w-5xl mx-auto px-6 py-16 space-y-12">
        <section className="space-y-5">
          <div className="inline-flex items-center gap-2 rounded-full bg-indigo-500/10 ring-1 ring-inset ring-indigo-400/30 px-4 py-1 text-[11px] tracking-wide uppercase font-medium text-indigo-300">Evidence Insights</div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">Common Health Myths</h1>
          <p className="text-sm text-slate-400 max-w-2xl leading-relaxed">Concise myth-busting with practical context. Always seek professional evaluation for personal medical decisions.</p>
        </section>
        <div className="grid gap-6 md:grid-cols-2">
          {myths.map((m, i) => (
            <div key={i} className="group relative rounded-lg border border-slate-700/60 bg-slate-850/60 backdrop-blur px-6 py-5 hover:border-indigo-400/50 transition-colors">
              <div className="absolute inset-px rounded-[10px] bg-gradient-to-br from-indigo-500/0 via-indigo-500/0 to-indigo-500/0 opacity-0 group-hover:opacity-100 group-hover:via-indigo-500/5 group-hover:to-indigo-500/10 transition-opacity pointer-events-none" />
              <p className="relative font-medium text-base leading-snug mb-2 text-slate-100">{m.myth}</p>
              <p className="relative text-[13px] text-slate-400 leading-relaxed">{m.fact}</p>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-slate-500">Disclaimer: Educational only. Not a substitute for professional diagnosis or treatment.</p>
      </main>
    </div>
  )
}
