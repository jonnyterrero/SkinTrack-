"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Save } from "lucide-react"
import { BODY_AREA_DEFINITIONS } from "@/lib/domain/body-areas"
import { SKIN_CONDITIONS } from "@/lib/domain/conditions"
import { SLEEP_HOURS } from "@/lib/domain/scales"
import type { NewSymptomRecordInput } from "@/lib/types"
import type { SeverityLevel } from "@/lib/types"
import { useSkinTrack } from "@/components/skintrack-provider"

type Props = {
  onRecordSaved: (input: NewSymptomRecordInput) => void | Promise<void>
}

export default function SymptomTracker({ onRecordSaved }: Props) {
  const { repository } = useSkinTrack()
  const catalog = repository.getMedicationCatalog().filter((m) => m.name.trim().length > 0)

  const [formData, setFormData] = useState({
    lesionLabel: "",
    condition: "",
    itch: [5],
    pain: [5],
    sleep: 8,
    stress: [5],
    triggers: "",
    newProducts: "",
    medications: "",
    adherence: false,
    notes: "",
    bodyArea: "",
    severity: "" as "" | SeverityLevel,
    selectedMedIds: [] as string[],
  })

  const toggleMed = (id: string) => {
    setFormData((prev) => ({
      ...prev,
      selectedMedIds: prev.selectedMedIds.includes(id)
        ? prev.selectedMedIds.filter((x) => x !== id)
        : [...prev.selectedMedIds, id],
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    const medNames = catalog
      .filter((m) => formData.selectedMedIds.includes(m.id))
      .map((m) => `${m.name}${m.dose ? ` (${m.dose})` : ""}`)
    const medicationsLine = [formData.medications.trim(), medNames.join(", ")].filter(Boolean).join(" · ")

    const input: NewSymptomRecordInput = {
      lesionLabel: formData.lesionLabel,
      condition: formData.condition,
      itch: formData.itch[0],
      pain: formData.pain[0],
      sleep: formData.sleep,
      stress: formData.stress[0],
      triggers: formData.triggers,
      newProducts: formData.newProducts,
      medications: medicationsLine,
      adherence: formData.adherence,
      notes: formData.notes,
      type: "symptom",
      ...(formData.bodyArea ? { bodyArea: formData.bodyArea } : {}),
      ...(formData.severity ? { severity: formData.severity } : {}),
      ...(formData.selectedMedIds.length ? { medicationIds: formData.selectedMedIds } : {}),
    }

    void onRecordSaved(input)

    setFormData({
      lesionLabel: "",
      condition: "",
      itch: [5],
      pain: [5],
      sleep: 8,
      stress: [5],
      triggers: "",
      newProducts: "",
      medications: "",
      adherence: false,
      notes: "",
      bodyArea: "",
      severity: "",
      selectedMedIds: [],
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Symptom Tracker</CardTitle>
        <CardDescription>Record your symptoms and track your skin condition progress</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="lesionLabel">Lesion Label</Label>
              <Input
                id="lesionLabel"
                placeholder="e.g., left forearm A"
                value={formData.lesionLabel}
                onChange={(e) => setFormData({ ...formData, lesionLabel: e.target.value })}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="condition">Condition Type</Label>
              <Select value={formData.condition} onValueChange={(value) => setFormData({ ...formData, condition: value })}>
                <SelectTrigger>
                  <SelectValue placeholder="Select condition" />
                </SelectTrigger>
                <SelectContent>
                  {SKIN_CONDITIONS.map((condition) => (
                    <SelectItem key={condition} value={condition}>
                      {condition}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="bodyArea">Body area</Label>
              <Select value={formData.bodyArea || "__none__"} onValueChange={(v) => setFormData({ ...formData, bodyArea: v === "__none__" ? "" : v })}>
                <SelectTrigger id="bodyArea">
                  <SelectValue placeholder="Select area (optional)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">Not specified</SelectItem>
                  {BODY_AREA_DEFINITIONS.map((a) => (
                    <SelectItem key={a.id} value={a.id}>
                      {a.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="severity">Severity (for map)</Label>
              <Select
                value={formData.severity || "__none__"}
                onValueChange={(v) => setFormData({ ...formData, severity: v === "__none__" ? "" : (v as SeverityLevel) })}
              >
                <SelectTrigger id="severity">
                  <SelectValue placeholder="Optional" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">Not specified</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Itch Level: {formData.itch[0]}/10</Label>
                <Slider
                  value={formData.itch}
                  onValueChange={(value) => setFormData({ ...formData, itch: value })}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <Label>Pain Level: {formData.pain[0]}/10</Label>
                <Slider
                  value={formData.pain}
                  onValueChange={(value) => setFormData({ ...formData, pain: value })}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="sleep">Sleep Hours Last Night</Label>
                <Input
                  id="sleep"
                  type="number"
                  min={SLEEP_HOURS.min}
                  max={SLEEP_HOURS.max}
                  step={SLEEP_HOURS.step}
                  value={formData.sleep}
                  onChange={(e) => setFormData({ ...formData, sleep: Number.parseFloat(e.target.value) })}
                />
              </div>

              <div className="space-y-2">
                <Label>Stress Level: {formData.stress[0]}/10</Label>
                <Slider
                  value={formData.stress}
                  onValueChange={(value) => setFormData({ ...formData, stress: value })}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="triggers">Triggers</Label>
              <Input
                id="triggers"
                placeholder="stress, sweat, fragrance"
                value={formData.triggers}
                onChange={(e) => setFormData({ ...formData, triggers: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="newProducts">New Products Used</Label>
              <Input
                id="newProducts"
                placeholder="new soap, lotion"
                value={formData.newProducts}
                onChange={(e) => setFormData({ ...formData, newProducts: e.target.value })}
              />
            </div>
          </div>

          {catalog.length > 0 ? (
            <div className="space-y-2">
              <Label>From your medication catalog</Label>
              <div className="flex flex-wrap gap-3 rounded-lg border border-slate-200/80 p-3 dark:border-slate-700">
                {catalog.map((m) => (
                  <div key={m.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={`med-${m.id}`}
                      checked={formData.selectedMedIds.includes(m.id)}
                      onCheckedChange={() => toggleMed(m.id)}
                    />
                    <Label htmlFor={`med-${m.id}`} className="cursor-pointer text-sm font-normal">
                      {m.name}
                      {m.dose ? <span className="text-muted-foreground"> ({m.dose})</span> : null}
                    </Label>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="space-y-2">
            <Label htmlFor="medications">Other medications / notes</Label>
            <Input
              id="medications"
              placeholder="Anything not in the catalog"
              value={formData.medications}
              onChange={(e) => setFormData({ ...formData, medications: e.target.value })}
            />
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="adherence"
              checked={formData.adherence}
              onCheckedChange={(checked) => setFormData({ ...formData, adherence: Boolean(checked) })}
            />
            <Label htmlFor="adherence">Took medications as planned</Label>
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">Additional Notes</Label>
            <Textarea
              id="notes"
              placeholder="Any other observations..."
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
            />
          </div>

          <Button type="submit" className="w-full">
            <Save className="w-4 h-4 mr-2" />
            Save Record
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
