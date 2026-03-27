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
import type { NewSymptomRecordInput } from "@/lib/types"

const CONDITIONS = [
  "eczema",
  "psoriasis",
  "guttate psoriasis",
  "keratosis pilaris",
  "cystic/hormonal acne",
  "melanoma",
  "vitiligo",
  "contact dermatitis",
  "cold sores",
]

type Props = {
  onRecordSaved: (input: NewSymptomRecordInput) => void | Promise<void>
}

export default function SymptomTracker({ onRecordSaved }: Props) {
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
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    const input: NewSymptomRecordInput = {
      lesionLabel: formData.lesionLabel,
      condition: formData.condition,
      itch: formData.itch[0],
      pain: formData.pain[0],
      sleep: formData.sleep,
      stress: formData.stress[0],
      triggers: formData.triggers,
      newProducts: formData.newProducts,
      medications: formData.medications,
      adherence: formData.adherence,
      notes: formData.notes,
      type: "symptom",
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
                  {CONDITIONS.map((condition) => (
                    <SelectItem key={condition} value={condition}>
                      {condition}
                    </SelectItem>
                  ))}
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
                  min={0}
                  max={24}
                  step={0.5}
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

          <div className="space-y-2">
            <Label htmlFor="medications">Medications Taken</Label>
            <Input
              id="medications"
              placeholder="triamcinolone, antihistamine"
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
