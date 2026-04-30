"use client"

import { useCallback, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { toast } from "sonner"
import { useAuth } from "@/context/AuthContext"
import { apiGet, apiSend } from "@/lib/api/client"
import type { MedCatalogItem, MedCategory } from "@/lib/types/backend"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

const CATEGORY_OPTIONS: { value: MedCategory; label: string }[] = [
  { value: "topical", label: "Topical prescription" },
  { value: "oral", label: "Oral medication" },
  { value: "injection", label: "Injection" },
  { value: "otc", label: "OTC medication" },
  { value: "moisturizer", label: "Moisturizer / emollient" },
  { value: "cleanser", label: "Cleanser / soap" },
  { value: "diet", label: "Diet change" },
  { value: "home_remedy", label: "Home remedy" },
  { value: "environmental", label: "Environmental change" },
  { value: "avoidance", label: "Avoidance / elimination" },
  { value: "other", label: "Other" },
]

type Draft = {
  name: string
  category: MedCategory
  dose: string
  frequency: string
  morning: boolean
  afternoon: boolean
  evening: boolean
  is_prescription: boolean
  prescribed_by: string
  notes: string
}

const emptyDraft = (): Draft => ({
  name: "",
  category: "topical",
  dose: "",
  frequency: "",
  morning: false,
  afternoon: false,
  evening: false,
  is_prescription: false,
  prescribed_by: "",
  notes: "",
})

export default function MedicationsPage() {
  const router = useRouter()
  const { user, loading } = useAuth()
  const [items, setItems] = useState<MedCatalogItem[]>([])
  const [draft, setDraft] = useState<Draft>(emptyDraft())
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/medications")
  }, [user, loading, router])

  const reload = useCallback(() => {
    apiGet<MedCatalogItem[]>("/api/medications")
      .then(setItems)
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (user) reload()
  }, [user, reload])

  async function add() {
    if (!draft.name.trim()) {
      toast.error("Name is required.")
      return
    }
    setSaving(true)
    try {
      await apiSend("/api/medications", "POST", {
        name: draft.name.trim(),
        category: draft.category,
        dose: draft.dose.trim() || null,
        frequency: draft.frequency.trim() || null,
        morning: draft.morning,
        afternoon: draft.afternoon,
        evening: draft.evening,
        is_prescription: draft.is_prescription,
        prescribed_by: draft.prescribed_by.trim() || null,
        notes: draft.notes.trim() || null,
        active: true,
      })
      toast.success("Added.")
      setDraft(emptyDraft())
      reload()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Add failed")
    } finally {
      setSaving(false)
    }
  }

  async function toggleActive(m: MedCatalogItem) {
    try {
      await apiSend(`/api/medications/${m.id}`, "PATCH", { active: !m.active })
      reload()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Update failed")
    }
  }

  async function remove(m: MedCatalogItem) {
    if (!confirm(`Delete ${m.name}? This cannot be undone.`)) return
    try {
      await apiSend(`/api/medications/${m.id}`, "DELETE")
      reload()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Delete failed")
    }
  }

  if (loading || !user) return <div className="p-8 text-center">Loading…</div>

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-2xl space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Medications &amp; remedies</h1>
          <Link href="/" className="text-sm text-slate-600 underline">
            Back to app
          </Link>
        </div>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Add a medication or remedy</h2>
          <div>
            <Label htmlFor="medname">Name</Label>
            <Input
              id="medname"
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
              placeholder="e.g. Tacrolimus 0.1%, CeraVe cream"
            />
          </div>
          <div>
            <Label>Category</Label>
            <Select
              value={draft.category}
              onValueChange={(v) =>
                setDraft({ ...draft, category: v as MedCategory })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CATEGORY_OPTIONS.map((c) => (
                  <SelectItem key={c.value} value={c.value}>
                    {c.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label htmlFor="dose">Dose</Label>
              <Input
                id="dose"
                value={draft.dose}
                onChange={(e) => setDraft({ ...draft, dose: e.target.value })}
                placeholder="e.g. 1 pea-sized, 10 mg"
              />
            </div>
            <div>
              <Label htmlFor="freq">Frequency</Label>
              <Input
                id="freq"
                value={draft.frequency}
                onChange={(e) =>
                  setDraft({ ...draft, frequency: e.target.value })
                }
                placeholder="e.g. twice daily"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-4 text-sm">
            <label className="flex items-center gap-2">
              <Checkbox
                checked={draft.morning}
                onCheckedChange={(v) => setDraft({ ...draft, morning: !!v })}
              />
              Morning
            </label>
            <label className="flex items-center gap-2">
              <Checkbox
                checked={draft.afternoon}
                onCheckedChange={(v) => setDraft({ ...draft, afternoon: !!v })}
              />
              Afternoon
            </label>
            <label className="flex items-center gap-2">
              <Checkbox
                checked={draft.evening}
                onCheckedChange={(v) => setDraft({ ...draft, evening: !!v })}
              />
              Evening
            </label>
            <label className="flex items-center gap-2">
              <Checkbox
                checked={draft.is_prescription}
                onCheckedChange={(v) =>
                  setDraft({ ...draft, is_prescription: !!v })
                }
              />
              Prescription
            </label>
          </div>
          {draft.is_prescription && (
            <div>
              <Label htmlFor="pby">Prescribed by</Label>
              <Input
                id="pby"
                value={draft.prescribed_by}
                onChange={(e) =>
                  setDraft({ ...draft, prescribed_by: e.target.value })
                }
                placeholder="Dr. name / clinic"
              />
            </div>
          )}
          <div>
            <Label htmlFor="notes">Notes</Label>
            <Textarea
              id="notes"
              rows={2}
              value={draft.notes}
              onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
            />
          </div>
          <Button onClick={add} disabled={saving}>
            {saving ? "Adding…" : "Add"}
          </Button>
        </Card>

        <Card className="p-5 space-y-2">
          <h2 className="text-lg font-medium">Your list ({items.length})</h2>
          {items.length === 0 && (
            <p className="text-sm text-slate-500">Nothing added yet.</p>
          )}
          {items.map((m) => (
            <div
              key={m.id}
              className="flex items-center justify-between border-b border-slate-100 py-2 last:border-0"
            >
              <div>
                <div className="font-medium text-sm">
                  {m.name}{" "}
                  <span className="text-xs text-slate-500">
                    · {m.category.replace("_", " ")}
                  </span>
                  {!m.active && (
                    <span className="ml-2 text-xs text-amber-600">
                      (inactive)
                    </span>
                  )}
                </div>
                <div className="text-xs text-slate-600">
                  {[m.dose, m.frequency].filter(Boolean).join(" · ")}
                </div>
                <div className="text-xs text-slate-500">
                  {[
                    m.morning && "AM",
                    m.afternoon && "noon",
                    m.evening && "PM",
                    m.is_prescription && "Rx",
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </div>
              </div>
              <div className="flex gap-1">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => toggleActive(m)}
                >
                  {m.active ? "Archive" : "Restore"}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => remove(m)}
                >
                  Delete
                </Button>
              </div>
            </div>
          ))}
        </Card>
      </div>
    </div>
  )
}
