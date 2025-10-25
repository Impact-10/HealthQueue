import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"
import { LandingHeader } from "@/components/layout/landing-header"

// Placeholder article + discovery data
interface Article {
  id: string
  title: string
  category: string
  minutes: number
  excerpt: string
  image: string
  impact?: string
}

interface Discovery {
  id: string
  title: string
  impact: "High" | "Critical" | "Moderate"
  timeAgo: string
}

const featured: Article = {
  id: "feat-1",
  title: "Revolutionary Gene Therapy Shows 85% Success Rate in Treating Rare Blood Disorders",
  category: "Medical Research",
  minutes: 8,
  excerpt: "Clinical trials demonstrate unprecedented effectiveness in treating beta-thalassemia and sickle cell disease, offering hope to millions of patients worldwide with previously incurable conditions.",
  image: "/placeholder.svg?height=420&width=960",
  impact: "High"
}

const articles: Article[] = [
  {
    id: "a1",
    title: "Daily 30-Minute Walks Reduce Heart Disease Risk by 40%...",
    category: "Prevention",
    minutes: 5,
    excerpt: "Large-scale research involving 50,000 participants confirms that moderate daily exercise significantly improves...",
    image: "/placeholder.svg?height=220&width=380"
  },
  {
    id: "a2",
    title: "Breakthrough AI System Detects Alzheimer's Disease 15 Years...",
    category: "Diagnostics",
    minutes: 7,
    excerpt: "Advanced machine learning algorithm analyzes brain scans and biomarkers to identify early signs of dementia,...",
    image: "/placeholder.svg?height=220&width=380"
  },
  {
    id: "a3",
    title: "Mindfulness Meditation Proven to Reduce Anxiety and Depression b...",
    category: "Mental Health",
    minutes: 6,
    excerpt: "Meta-analysis of 200 studies confirms the therapeutic benefits of meditation practices for mental health, with effect...",
    image: "/placeholder.svg?height=220&width=380"
  }
]

const discoveries: Discovery[] = [
  { id: "d1", title: "Gene therapy breakthrough for rare diseases", impact: "High", timeAgo: "2h ago" },
  { id: "d2", title: "New Alzheimer's drug shows 30% improvement", impact: "Critical", timeAgo: "4h ago" },
  { id: "d3", title: "AI detects cancer 95% accuracy in trials", impact: "High", timeAgo: "6h ago" },
  { id: "d4", title: "Stem cell treatment restores vision", impact: "Moderate", timeAgo: "8h ago" },
  { id: "d5", title: "Vaccine development for autoimmune disorders", impact: "High", timeAgo: "12h ago" },
]

function impactColor(impact?: string) {
  switch (impact) {
    case "Critical":
      return "text-rose-600 dark:text-rose-400"
    case "High":
      return "text-amber-600 dark:text-amber-400"
    case "Moderate":
      return "text-blue-600 dark:text-blue-400"
    default:
      return "text-muted-foreground"
  }
}

export const dynamic = "force-dynamic"
export const metadata = { title: 'Home' }

export default function HealthHomePage() {
  return (
    <div className="relative min-h-screen bg-gradient-to-b from-slate-950 via-slate-930 to-slate-900 text-slate-200 pb-20">
      <LandingHeader />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-12 space-y-14">
        {/* Hero: AI Chat focus */}
        <section className="relative overflow-hidden rounded-2xl border border-slate-700/70 bg-gradient-to-br from-slate-900/70 via-slate-900/40 to-indigo-900/30 backdrop-blur px-8 py-14 flex flex-col lg:flex-row gap-12">
          <div className="flex-1 space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full bg-indigo-500/15 ring-1 ring-inset ring-indigo-400/30 px-4 py-1 text-[11px] tracking-wide uppercase font-medium text-indigo-300">AI Health Companion</div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-tight max-w-2xl bg-gradient-to-r from-slate-50 via-slate-200 to-indigo-200 bg-clip-text text-transparent">Chat with an Intelligent Health Assistant</h1>
            <p className="text-sm md:text-base text-slate-400 max-w-xl leading-relaxed">Ask evidence-aware questions, get lifestyle guidance, and generate structured consultation summaries. Your data is protected and you always get a reminder to seek real medical care for serious concerns.</p>
            <div className="flex flex-wrap gap-3 pt-2">
              <a href="/auth/login" className="text-[13px] font-medium px-5 py-2.5 rounded-md bg-indigo-500 text-white shadow hover:bg-indigo-400 transition-colors">Start Chatting</a>
              <a href="/myths" className="text-[13px] font-medium px-5 py-2.5 rounded-md border border-slate-600/60 text-slate-300 hover:bg-slate-800/60 transition-colors">Learn Myths</a>
              <a href="/first-aid" className="text-[13px] font-medium px-5 py-2.5 rounded-md border border-slate-600/60 text-slate-300 hover:bg-slate-800/60 transition-colors">First Aid Basics</a>
            </div>
            <ul className="grid sm:grid-cols-3 gap-4 pt-6 text-[11px] text-slate-400 max-w-2xl">
              <li className="flex items-start gap-2"><span className="mt-1 size-1.5 rounded-full bg-emerald-400" />No general chit‑chat: focused health context</li>
              <li className="flex items-start gap-2"><span className="mt-1 size-1.5 rounded-full bg-amber-400" />Encourages professional follow‑up</li>
              <li className="flex items-start gap-2"><span className="mt-1 size-1.5 rounded-full bg-blue-400" />Generates conversation summaries</li>
            </ul>
          </div>
          <div className="flex-1 relative min-h-[320px] lg:min-h-[380px]">
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-indigo-500/20 via-purple-500/10 to-transparent blur-3xl" />
            <div className="relative h-full w-full grid grid-rows-6 grid-cols-6 gap-2">
              {[...Array(16)].map((_,i) => (
                <div key={i} className="col-span-2 row-span-2 rounded-lg border border-slate-700/60 bg-slate-800/40 flex items-center justify-center text-[10px] text-slate-400">
                  {i % 3 === 0 ? 'Sleep' : i % 3 === 1 ? 'Nutrition' : 'Stress'}
                </div>
              ))}
            </div>
          </div>
          <div className="absolute -top-20 -right-20 w-80 h-80 bg-indigo-600/20 rounded-full blur-3xl pointer-events-none" />
        </section>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-10 pt-4">
          <section className="lg:col-span-8 space-y-10">
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 rounded-full bg-indigo-500/10 ring-1 ring-inset ring-indigo-400/30 px-4 py-1 text-[11px] tracking-wide uppercase font-medium text-indigo-300">Live Insights</div>
              <h2 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">Breaking Health Facts</h2>
              <p className="text-sm text-slate-400 max-w-2xl leading-relaxed">Curated emerging medical and wellness intelligence. Contextual, concise, and continuously evolving.</p>
              <FeaturedArticle article={featured} />
            </div>
            <div className="grid gap-7 md:grid-cols-2 xl:grid-cols-3">
              {articles.map(a => <ArticleCard key={a.id} article={a} />)}
            </div>
          </section>
          <aside className="lg:col-span-4 space-y-8">
            <LatestDiscoveries />
            <HealthRiskZones />
          </aside>
        </main>
      </div>
    </div>
  )
}


function FeaturedArticle({ article }: { article: Article }) {
  return (
    <Card className="overflow-hidden border-slate-700/70 bg-slate-800/40 backdrop-blur supports-[backdrop-filter]:bg-slate-800/30 shadow-sm relative">
      <div className="relative h-64 md:h-80 w-full">
        <Image src={article.image} alt={article.title} fill className="object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/90 via-slate-900/40 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 p-7 space-y-4">
          <div className="flex items-center gap-3 text-[11px] font-medium">
            <Badge variant="secondary" className="uppercase tracking-wide bg-indigo-500/20 text-indigo-300 ring-1 ring-inset ring-indigo-400/30">{article.category}</Badge>
            <span className="text-slate-400">{article.minutes} min read</span>
            {article.impact && <span className="text-amber-300 font-medium">{article.impact} Impact</span>}
          </div>
            <h3 className="text-2xl md:text-3xl font-semibold leading-snug max-w-3xl text-slate-100">{article.title}</h3>
            <p className="text-sm text-slate-400/90 max-w-2xl leading-relaxed line-clamp-3">{article.excerpt}</p>
        </div>
      </div>
    </Card>
  )
}

function ArticleCard({ article }: { article: Article }) {
  return (
    <Card className="overflow-hidden group border-slate-700/60 hover:border-indigo-400/50 transition-colors bg-slate-800/40 backdrop-blur supports-[backdrop-filter]:bg-slate-800/30">
      <div className="h-40 w-full relative">
        <Image src={article.image} alt={article.title} fill className="object-cover" />
        <div className="absolute inset-0 bg-slate-950/0 group-hover:bg-slate-950/20 transition-colors" />
      </div>
      <CardContent className="p-5 space-y-3">
        <div className="flex items-center gap-2 text-[11px]">
          <Badge variant="outline" className="font-medium uppercase tracking-wide rounded-sm px-2 py-0.5 text-[10px] border-slate-600/60 text-slate-300">{article.category}</Badge>
          <span className="text-slate-500">{article.minutes} min</span>
        </div>
        <h4 className="font-semibold text-[13px] leading-snug line-clamp-2 text-slate-100">{article.title}</h4>
        <p className="text-[12px] text-slate-400 line-clamp-3 leading-relaxed">{article.excerpt}</p>
      </CardContent>
    </Card>
  )
}

function LatestDiscoveries() {
  return (
    <Card className="border-slate-700/70 bg-slate-800/40 backdrop-blur supports-[backdrop-filter]:bg-slate-800/30">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2 text-slate-100">🔬 Latest Discoveries</CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        {discoveries.map(d => (
          <button key={d.id} className="w-full text-left group rounded-md px-2 py-2 hover:bg-slate-700/40 transition flex items-start gap-3">
            <div className="flex-1 space-y-1">
              <p className="text-[13px] font-medium leading-snug group-hover:text-slate-100 line-clamp-2 text-slate-300">{d.title}</p>
              <div className="flex items-center gap-2 text-[10px]">
                <span className={impactColor(d.impact) + ' font-medium'}>{d.impact} Impact</span>
                <span className="text-slate-500">{d.timeAgo}</span>
              </div>
            </div>
            <span className="text-slate-600 group-hover:text-slate-400">→</span>
          </button>
        ))}
      </CardContent>
    </Card>
  )
}

function HealthRiskZones() {
  return (
    <Card className="border-slate-700/70 bg-slate-800/40 backdrop-blur supports-[backdrop-filter]:bg-slate-800/30">
      <CardHeader className="pb-3"><CardTitle className="text-sm flex items-center gap-2 text-slate-100">⚠️ Health Risk Zones</CardTitle></CardHeader>
      <CardContent>
        <div className="aspect-[4/3] w-full rounded-md bg-gradient-to-br from-rose-500/15 via-amber-500/15 to-blue-500/15 flex items-center justify-center">
          <div className="grid grid-cols-3 gap-6">
            <div className="size-3 rounded-full bg-rose-400/80 animate-pulse" />
            <div className="size-3 rounded-full bg-amber-400/80 animate-pulse delay-150" />
            <div className="size-3 rounded-full bg-blue-400/80 animate-pulse delay-300" />
            <div className="size-3 rounded-full bg-rose-400/80 animate-pulse delay-300" />
            <div className="size-3 rounded-full bg-amber-400/80 animate-pulse" />
            <div className="size-3 rounded-full bg-blue-400/80 animate-pulse delay-150" />
          </div>
        </div>
        <p className="mt-3 text-[11px] text-slate-500 leading-relaxed">Prototype visualization illustrating relative risk intensity across monitored geo-cohorts. Replace with real epidemiological overlays or environmental exposure datasets.</p>
      </CardContent>
    </Card>
  )
}
