"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Loader2, FileText, Copy, Download } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import dynamic from "next/dynamic"

const MarkdownContent = dynamic(() => import("./markdown-content").then(m => m.MarkdownContent), { ssr: false })

interface ChatReportDialogProps {
  threadId: string
  userId: string
  disabled?: boolean
}

export function ChatReportDialog({ threadId, userId, disabled }: ChatReportDialogProps) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState<string | null>(null)
  const { toast } = useToast()

  const generate = async () => {
    if (!threadId) return
    setLoading(true)
    setReport(null)
    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threadId, userId }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || "Failed to generate report")
      setReport(data.report)
    } catch (e: any) {
      toast({ title: "Report error", description: e.message, variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  const copyReport = async () => {
    if (!report) return
    try {
      await navigator.clipboard.writeText(report)
      toast({ title: "Copied", description: "Report copied to clipboard" })
    } catch {}
  }

  const downloadReport = () => {
    if (!report) return
    const blob = new Blob([report], { type: "text/markdown" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `health-report-${threadId}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Dialog open={open} onOpenChange={(o) => { setOpen(o); if (o && !report && !loading && threadId) generate() }}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={!threadId || disabled} className="gap-1"> <FileText className="h-4 w-4"/> Report </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto bg-background/95 backdrop-blur">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2"><FileText className="h-5 w-5 text-primary"/> Diagnostic Report</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {loading && (
            <div className="flex items-center gap-2 text-muted-foreground text-sm"><Loader2 className="h-4 w-4 animate-spin"/> Generating summary…</div>
          )}
          {!loading && report && (
            <div className="space-y-4">
              <div className="prose prose-invert max-w-none text-sm leading-relaxed">
                <MarkdownContent content={report} />
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={copyReport} className="gap-1"><Copy className="h-4 w-4"/> Copy</Button>
                <Button size="sm" variant="outline" onClick={downloadReport} className="gap-1"><Download className="h-4 w-4"/> Download</Button>
                <Button size="sm" variant="ghost" onClick={() => generate()} className="gap-1"><Loader2 className="h-4 w-4"/> Regenerate</Button>
              </div>
            </div>
          )}
          {!loading && !report && (
            <div className="text-xs text-muted-foreground">No report yet. It will generate automatically.</div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
