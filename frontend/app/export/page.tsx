"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { toast } from "sonner"
import { useAuth } from "@/context/AuthContext"
import { apiGet, apiSend } from "@/lib/api/client"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type LesionRow = { id: string; label: string }
type SkinEventRow = {
  id: string
  ts: string
  lesion_id: string
  severity_0_4: number
  itch: number
  pain: number
  burning: number
  dryness: number
  stress: number
  sleep_hours: number
  sleep_quality: number
  notes: string | null
}

function toCsv(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return ""
  const headers = Array.from(
    rows.reduce((set, r) => {
      Object.keys(r).forEach((k) => set.add(k))
      return set
    }, new Set<string>()),
  )
  const escape = (v: unknown) => {
    if (v == null) return ""
    const s = typeof v === "string" ? v : JSON.stringify(v)
    return /[,"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return [
    headers.join(","),
    ...rows.map((r) => headers.map((h) => escape(r[h])).join(",")),
  ].join("\n")
}

function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function ExportPage() {
  const router = useRouter()
  const { user, loading } = useAuth()
  const [lesions, setLesions] = useState<LesionRow[]>([])
  const [lesionId, setLesionId] = useState<string>("all")
  const [dateFrom, setDateFrom] = useState("")
  const [dateTo, setDateTo] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/export")
  }, [user, loading, router])

  useEffect(() => {
    if (!user) return
    apiGet<LesionRow[]>("/api/lesions").then(setLesions).catch(() => {})
  }, [user])

  async function fetchEvents(): Promise<SkinEventRow[]> {
    const rows = await apiGet<SkinEventRow[]>("/api/skin-events?limit=1000")
    return rows.filter((e) => {
      if (lesionId !== "all" && e.lesion_id !== lesionId) return false
      const ts = e.ts.slice(0, 10)
      if (dateFrom && ts < dateFrom) return false
      if (dateTo && ts > dateTo) return false
      return true
    })
  }

  async function exportCsv() {
    setBusy(true)
    try {
      const events = await fetchEvents()
      if (events.length === 0) {
        toast.info("No events match this filter.")
        return
      }
      const csv = toCsv(events)
      download(
        `skintrack-${lesionId === "all" ? "all" : "lesion"}-${new Date()
          .toISOString()
          .slice(0, 10)}.csv`,
        csv,
        "text/csv;charset=utf-8",
      )
      await apiSend("/api/exports", "POST", {
        lesion_id: lesionId === "all" ? null : lesionId,
        export_type: lesionId === "all" ? "csv_full" : "csv_lesion",
        date_from: dateFrom || null,
        date_to: dateTo || null,
      })
      toast.success(`Exported ${events.length} rows.`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Export failed")
    } finally {
      setBusy(false)
    }
  }

  async function exportJson() {
    setBusy(true)
    try {
      const full = await apiGet<unknown>("/api/account")
      download(
        `skintrack-full-${new Date().toISOString().slice(0, 10)}.json`,
        JSON.stringify(full, null, 2),
        "application/json",
      )
      await apiSend("/api/exports", "POST", { export_type: "json_full" })
      toast.success("Full export downloaded.")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Export failed")
    } finally {
      setBusy(false)
    }
  }

  async function openSummary() {
    const q = new URLSearchParams()
    if (lesionId !== "all") q.set("lesion", lesionId)
    if (dateFrom) q.set("from", dateFrom)
    if (dateTo) q.set("to", dateTo)
    await apiSend("/api/exports", "POST", {
      lesion_id: lesionId === "all" ? null : lesionId,
      export_type: lesionId === "all" ? "pdf_summary" : "pdf_lesion",
      date_from: dateFrom || null,
      date_to: dateTo || null,
    })
    window.open(`/export/summary?${q.toString()}`, "_blank")
  }

  if (loading || !user) return <div className="p-8 text-center">Loading…</div>

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-xl space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Export center</h1>
          <Link href="/" className="text-sm text-slate-600 underline">
            Back
          </Link>
        </div>

        <Card className="p-5 space-y-3">
          <div>
            <Label>Lesion filter</Label>
            <Select value={lesionId} onValueChange={setLesionId}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All lesions</SelectItem>
                {lesions.map((l) => (
                  <SelectItem key={l.id} value={l.id}>
                    {l.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label htmlFor="from">From</Label>
              <Input
                id="from"
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="to">To</Label>
              <Input
                id="to"
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button onClick={exportCsv} disabled={busy}>
              Download CSV
            </Button>
            <Button onClick={openSummary} variant="outline" disabled={busy}>
              Clinician summary (PDF)
            </Button>
            <Button onClick={exportJson} variant="ghost" disabled={busy}>
              Full JSON backup
            </Button>
          </div>
          <p className="text-xs text-slate-500">
            The clinician summary opens a printable page. Use your browser&apos;s
            Print → Save as PDF to save or share.
          </p>
        </Card>
      </div>
    </div>
  )
}
