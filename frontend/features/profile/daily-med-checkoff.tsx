"use client"

import { useEffect, useMemo, useState } from "react"
import { format } from "date-fns"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useSkinTrack } from "@/components/skintrack-provider"
import type { DailyMedCheckoff, MedAdherenceStatus, MedicationCatalogItem } from "@/lib/domain/medications"

function todayYmd(): string {
  return format(new Date(), "yyyy-MM-dd")
}

export default function DailyMedCheckoff() {
  const { repository, records } = useSkinTrack()
  const [date, setDate] = useState(todayYmd)
  const [catalog, setCatalog] = useState<MedicationCatalogItem[]>([])
  const [map, setMap] = useState<Record<string, DailyMedCheckoff>>({})

  useEffect(() => {
    setCatalog(repository.getMedicationCatalog())
    setMap(repository.getMedDailyByDate())
  }, [repository, records])

  const checkoff = useMemo(() => map[date] ?? { date, byMedicationId: {} }, [map, date])

  const setStatus = (medId: string, value: string) => {
    const rest = { ...checkoff.byMedicationId }
    if (value === "__unset__") {
      delete rest[medId]
    } else {
      rest[medId] = value as MedAdherenceStatus
    }
    const next: DailyMedCheckoff = { date, byMedicationId: rest }
    const merged = { ...map, [date]: next }
    setMap(merged)
    repository.setMedDailyByDate(merged)
  }

  const activeMeds = catalog.filter((m) => m.name.trim().length > 0)

  if (activeMeds.length === 0) {
    return (
      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Daily medication checkoff</CardTitle>
          <CardDescription>Add medications in the catalog above to track daily adherence.</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card className="glass-card border-slate-200/80 dark:border-slate-700">
      <CardHeader>
        <CardTitle className="text-lg">Daily medication checkoff</CardTitle>
        <CardDescription>Mark each medication for the selected day</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="checkoff-date">Date</Label>
          <InputDate value={date} onChange={setDate} id="checkoff-date" />
        </div>
        <div className="space-y-3">
          {activeMeds.map((m) => {
            const current = checkoff.byMedicationId[m.id]
            return (
              <div
                key={m.id}
                className="flex flex-col gap-2 rounded-lg border border-slate-200/80 p-3 sm:flex-row sm:items-center sm:justify-between dark:border-slate-700"
              >
                <div>
                  <div className="font-medium">{m.name}</div>
                  {m.dose ? <div className="text-xs text-muted-foreground">{m.dose}</div> : null}
                </div>
                <Select
                  value={current ?? "__unset__"}
                  onValueChange={(v) => setStatus(m.id, v as MedAdherenceStatus)}
                >
                  <SelectTrigger className="w-full sm:w-[180px]">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__unset__">Not set</SelectItem>
                    <SelectItem value="taken">Taken</SelectItem>
                    <SelectItem value="partial">Partial</SelectItem>
                    <SelectItem value="skipped">Skipped</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )
          })}
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => {
            setDate(todayYmd())
          }}
        >
          Jump to today
        </Button>
      </CardContent>
    </Card>
  )
}

function InputDate({ id, value, onChange }: { id?: string; value: string; onChange: (v: string) => void }) {
  return (
    <input
      id={id}
      type="date"
      className="glass-input flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}
