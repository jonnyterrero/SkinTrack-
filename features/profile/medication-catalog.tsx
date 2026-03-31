"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Textarea } from "@/components/ui/textarea"
import { Trash2 } from "lucide-react"
import { useSkinTrack } from "@/components/skintrack-provider"
import { emptyMedicationItem, newMedicationId, type MedicationCatalogItem } from "@/lib/domain/medications"

export default function MedicationCatalog() {
  const { repository, records } = useSkinTrack()
  const [items, setItems] = useState<MedicationCatalogItem[]>([])

  useEffect(() => {
    setItems(repository.getMedicationCatalog())
  }, [repository, records])

  const persist = (next: MedicationCatalogItem[]) => {
    setItems(next)
    repository.setMedicationCatalog(next)
  }

  const addRow = () => {
    persist([...items, { ...emptyMedicationItem(), id: newMedicationId() }])
  }

  const update = (id: string, patch: Partial<MedicationCatalogItem>) => {
    persist(items.map((it) => (it.id === id ? { ...it, ...patch } : it)))
  }

  const remove = (id: string) => {
    persist(items.filter((it) => it.id !== id))
  }

  return (
    <Card className="glass-card border-slate-200/80 dark:border-slate-700">
      <CardHeader>
        <CardTitle className="text-lg">Medication catalog</CardTitle>
        <CardDescription>Reusable medications for daily checkoff and symptom logging</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">No medications yet. Add one to enable structured tracking.</p>
        ) : null}
        <div className="space-y-4">
          {items.map((m) => (
            <div key={m.id} className="space-y-3 rounded-xl border border-slate-200/80 p-4 dark:border-slate-700">
              <div className="flex flex-wrap gap-3 md:flex-nowrap">
                <div className="min-w-0 flex-1 space-y-1">
                  <Label>Name</Label>
                  <Input value={m.name} onChange={(e) => update(m.id, { name: e.target.value })} placeholder="e.g. Triamcinolone" />
                </div>
                <div className="w-full space-y-1 md:w-32">
                  <Label>Dose</Label>
                  <Input value={m.dose} onChange={(e) => update(m.id, { dose: e.target.value })} placeholder="0.1%" />
                </div>
                <Button type="button" variant="ghost" size="icon" className="mt-6 shrink-0" onClick={() => remove(m.id)} aria-label="Remove">
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <Checkbox id={`${m.id}-morning`} checked={m.morning} onCheckedChange={(c) => update(m.id, { morning: Boolean(c) })} />
                  <Label htmlFor={`${m.id}-morning`} className="font-normal">
                    Morning
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox id={`${m.id}-aft`} checked={m.afternoon} onCheckedChange={(c) => update(m.id, { afternoon: Boolean(c) })} />
                  <Label htmlFor={`${m.id}-aft`} className="font-normal">
                    Afternoon
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox id={`${m.id}-eve`} checked={m.evening} onCheckedChange={(c) => update(m.id, { evening: Boolean(c) })} />
                  <Label htmlFor={`${m.id}-eve`} className="font-normal">
                    Evening
                  </Label>
                </div>
              </div>
              <div className="space-y-1">
                <Label>Notes</Label>
                <Textarea value={m.notes} onChange={(e) => update(m.id, { notes: e.target.value })} rows={2} className="resize-none" />
              </div>
            </div>
          ))}
        </div>
        <Button type="button" variant="outline" onClick={addRow} className="w-full">
          Add medication
        </Button>
      </CardContent>
    </Card>
  )
}
