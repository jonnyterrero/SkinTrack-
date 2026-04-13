"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useSkinTrack } from "@/components/skintrack-provider"
import {
  LESION_SEVERITY_0_4,
  newLesionId,
  SKIN_LOCATION_OPTIONS,
  SLEEP_HOURS_RANGE,
  SLEEP_QUALITY,
  SYMPTOM_INTENSITY,
} from "@/lib/domain/skin-event-metrics"
import type { NewSkinEventInput } from "@/lib/types"

export default function SkinEventLog() {
  const { lesions, upsertLesion, saveRecord } = useSkinTrack()

  const activeLesions = useMemo(() => lesions.filter((l) => !l.archived), [lesions])

  const [newLesionLabel, setNewLesionLabel] = useState("")
  const [lesionId, setLesionId] = useState<string>("")
  const [severity04, setSeverity04] = useState<0 | 1 | 2 | 3 | 4>(2)
  const [locationId, setLocationId] = useState<string>(SKIN_LOCATION_OPTIONS[0]?.id ?? "face.forehead")
  const [itch, setItch] = useState([5])
  const [pain, setPain] = useState([5])
  const [burning, setBurning] = useState([5])
  const [dryness, setDryness] = useState([5])
  const [stress, setStress] = useState([5])
  const [sleepHours, setSleepHours] = useState(8)
  const [sleepQuality, setSleepQuality] = useState<1 | 2 | 3 | 4 | 5>(3)
  const [notes, setNotes] = useState("")
  const [saving, setSaving] = useState(false)

  const createLesion = () => {
    const label = newLesionLabel.trim()
    if (!label) return
    const id = newLesionId()
    upsertLesion({ id, label, createdAt: new Date().toISOString() })
    setLesionId(id)
    setNewLesionLabel("")
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!lesionId) return
    setSaving(true)
    const input: NewSkinEventInput = {
      type: "skin_event",
      lesionId,
      severity04,
      locationId,
      itch: itch[0] ?? 0,
      pain: pain[0] ?? 0,
      burning: burning[0] ?? 0,
      dryness: dryness[0] ?? 0,
      stress: stress[0] ?? 0,
      sleepHours,
      sleepQuality,
      notes: notes.trim() || undefined,
    }
    const ok = await saveRecord(input)
    setSaving(false)
    if (ok) {
      setNotes("")
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Skin event (v1 metrics)</CardTitle>
        <CardDescription>
          Structured log for longitudinal tracking — lesion severity 0–4, location, intensities, sleep. Optional notes
          only.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={(e) => void submit(e)} className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Lesion</Label>
              <Select value={lesionId || "__none__"} onValueChange={(v) => setLesionId(v === "__none__" ? "" : v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select or create below" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">Not selected</SelectItem>
                  {activeLesions.map((l) => (
                    <SelectItem key={l.id} value={l.id}>
                      {l.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="new-lesion">New lesion name</Label>
              <div className="flex gap-2">
                <Input
                  id="new-lesion"
                  value={newLesionLabel}
                  onChange={(e) => setNewLesionLabel(e.target.value)}
                  placeholder="e.g. Left forearm patch"
                />
                <Button type="button" variant="secondary" onClick={createLesion}>
                  Create
                </Button>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Lesion severity (0–4)</Label>
            <div className="flex flex-wrap gap-2">
              {LESION_SEVERITY_0_4.map((s) => (
                <Button
                  key={s.value}
                  type="button"
                  size="sm"
                  variant={severity04 === s.value ? "default" : "outline"}
                  onClick={() => setSeverity04(s.value)}
                  title={s.description}
                >
                  {s.value} — {s.label}
                </Button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">{LESION_SEVERITY_0_4.find((x) => x.value === severity04)?.description}</p>
          </div>

          <div className="space-y-2">
            <Label>Location</Label>
            <Select value={locationId} onValueChange={setLocationId}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SKIN_LOCATION_OPTIONS.map((opt) => (
                  <SelectItem key={opt.id} value={opt.id}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <SliderRow label="Itch" value={itch} onChange={setItch} />
            <SliderRow label="Pain" value={pain} onChange={setPain} />
            <SliderRow label="Burning" value={burning} onChange={setBurning} />
            <SliderRow label="Dryness" value={dryness} onChange={setDryness} />
            <SliderRow label="Stress" value={stress} onChange={setStress} />
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="sleep-h">Sleep hours</Label>
              <Input
                id="sleep-h"
                type="number"
                min={SLEEP_HOURS_RANGE.min}
                max={SLEEP_HOURS_RANGE.max}
                step={SLEEP_HOURS_RANGE.step}
                value={sleepHours}
                onChange={(e) => setSleepHours(Number.parseFloat(e.target.value) || 0)}
              />
            </div>
            <div className="space-y-2">
              <Label>Sleep quality (1–5)</Label>
              <Select
                value={String(sleepQuality)}
                onValueChange={(v) => setSleepQuality(Number.parseInt(v, 10) as 1 | 2 | 3 | 4 | 5)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[1, 2, 3, 4, 5].map((n) => (
                    <SelectItem key={n} value={String(n)}>
                      {n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">Notes (optional)</Label>
            <Textarea id="notes" rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} className="resize-none" />
          </div>

          <Button type="submit" className="w-full" disabled={!lesionId || saving}>
            {saving ? "Saving…" : "Save skin event"}
          </Button>
          <p className="text-center text-xs text-muted-foreground">
            Scales: symptom intensity {SYMPTOM_INTENSITY.min}–{SYMPTOM_INTENSITY.max}; {SLEEP_QUALITY.label} {SLEEP_QUALITY.min}–{SLEEP_QUALITY.max}.
          </p>
        </form>
      </CardContent>
    </Card>
  )
}

function SliderRow({
  label,
  value,
  onChange,
}: {
  label: string
  value: number[]
  onChange: (v: number[]) => void
}) {
  return (
    <div className="space-y-2">
      <Label>
        {label}: {value[0] ?? 0}/{SYMPTOM_INTENSITY.max}
      </Label>
      <Slider value={value} min={SYMPTOM_INTENSITY.min} max={SYMPTOM_INTENSITY.max} step={SYMPTOM_INTENSITY.step} onValueChange={onChange} />
    </div>
  )
}
