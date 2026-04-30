"use client"

import { Suspense, useCallback, useEffect, useMemo, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import { toast } from "sonner"
import { useAuth } from "@/context/AuthContext"
import { apiGet, apiSend } from "@/lib/api/client"
import type {
  EventMedication,
  EventProduct,
  EventTrigger,
  MedCatalogItem,
  MedCategory,
  TriggerTaxonomyRow,
} from "@/lib/types/backend"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

const PRODUCT_TYPE_OPTIONS: { value: MedCategory; label: string }[] = [
  { value: "moisturizer", label: "Moisturizer" },
  { value: "cleanser", label: "Cleanser" },
  { value: "topical", label: "Topical (Rx/OTC)" },
  { value: "oral", label: "Oral" },
  { value: "injection", label: "Injection" },
  { value: "otc", label: "OTC" },
  { value: "diet", label: "Diet change" },
  { value: "home_remedy", label: "Home remedy" },
  { value: "environmental", label: "Environmental change" },
  { value: "avoidance", label: "Avoidance / elimination" },
  { value: "other", label: "Other" },
]

function groupForKey(key: string): string {
  if (
    ["food", "dairy", "gluten", "sugar", "alcohol", "caffeine", "spicy"].includes(
      key,
    )
  ) {
    return "Diet"
  }
  if (["stress", "poor_sleep", "menstrual_cycle", "illness"].includes(key)) {
    return "Lifestyle"
  }
  if (["heat", "cold", "humidity", "dry_air", "sun", "sweat"].includes(key)) {
    return "Environment"
  }
  return "Contact / other"
}

export default function CheckInPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center">Loading…</div>}>
      <CheckInInner />
    </Suspense>
  )
}

function CheckInInner() {
  const router = useRouter()
  const params = useSearchParams()
  const eventId = params.get("event")
  const { user, loading } = useAuth()

  const [taxonomy, setTaxonomy] = useState<TriggerTaxonomyRow[]>([])
  const [meds, setMeds] = useState<MedCatalogItem[]>([])
  const [eventMeds, setEventMeds] = useState<EventMedication[]>([])
  const [triggers, setTriggers] = useState<EventTrigger[]>([])
  const [products, setProducts] = useState<EventProduct[]>([])

  const [newProduct, setNewProduct] = useState({
    product_name: "",
    product_type: "moisturizer" as MedCategory,
    first_use: false,
    notes: "",
  })

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/checkin")
  }, [user, loading, router])

  const loadAll = useCallback(async () => {
    if (!eventId) return
    try {
      const [tax, medsList, emList, trigList, prodList] = await Promise.all([
        apiGet<TriggerTaxonomyRow[]>("/api/triggers"),
        apiGet<MedCatalogItem[]>("/api/medications?active=true"),
        apiGet<EventMedication[]>(`/api/event-medications?skin_event_id=${eventId}`),
        apiGet<EventTrigger[]>(`/api/event-triggers?skin_event_id=${eventId}`),
        apiGet<EventProduct[]>(`/api/event-products?skin_event_id=${eventId}`),
      ])
      setTaxonomy(tax)
      setMeds(medsList)
      setEventMeds(emList)
      setTriggers(trigList)
      setProducts(prodList)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not load")
    }
  }, [eventId])

  useEffect(() => {
    if (user) loadAll()
  }, [user, loadAll])

  const takenMap = useMemo(
    () => new Map(eventMeds.map((e) => [e.med_catalog_id, e])),
    [eventMeds],
  )
  const activeTriggers = useMemo(
    () => new Set(triggers.map((t) => t.trigger_key)),
    [triggers],
  )

  const triggerGroups = useMemo(() => {
    const map = new Map<string, TriggerTaxonomyRow[]>()
    for (const row of taxonomy) {
      const g = groupForKey(row.key)
      if (!map.has(g)) map.set(g, [])
      map.get(g)!.push(row)
    }
    return Array.from(map.entries())
  }, [taxonomy])

  async function toggleMed(m: MedCatalogItem, taken: boolean) {
    if (!eventId) return
    try {
      await apiSend("/api/event-medications", "POST", {
        skin_event_id: eventId,
        med_catalog_id: m.id,
        taken,
      })
      loadAll()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Update failed")
    }
  }

  async function toggleTrigger(key: string, active: boolean) {
    if (!eventId) return
    try {
      if (active) {
        await apiSend("/api/event-triggers", "POST", {
          skin_event_id: eventId,
          trigger_key: key,
        })
      } else {
        const match = triggers.find((t) => t.trigger_key === key)
        if (match) await apiSend(`/api/event-triggers/${match.id}`, "DELETE")
      }
      loadAll()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Update failed")
    }
  }

  async function addProduct() {
    if (!eventId || !newProduct.product_name.trim()) {
      toast.error("Product name is required.")
      return
    }
    try {
      await apiSend("/api/event-products", "POST", {
        skin_event_id: eventId,
        product_name: newProduct.product_name.trim(),
        product_type: newProduct.product_type,
        first_use: newProduct.first_use,
        used: true,
        notes: newProduct.notes.trim() || null,
      })
      setNewProduct({
        product_name: "",
        product_type: "moisturizer",
        first_use: false,
        notes: "",
      })
      loadAll()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Add failed")
    }
  }

  async function removeProduct(id: string) {
    try {
      await apiSend(`/api/event-products/${id}`, "DELETE")
      loadAll()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Delete failed")
    }
  }

  if (loading || !user) return <div className="p-8 text-center">Loading…</div>
  if (!eventId) {
    return (
      <div className="mx-auto max-w-xl p-8 space-y-3">
        <h1 className="text-xl font-semibold">Daily check-in</h1>
        <p className="text-sm text-slate-600">
          Open this page from a specific skin event.
        </p>
        <Link href="/" className="text-sm underline">
          Back
        </Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-2xl space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Daily check-in</h1>
          <Link href="/" className="text-sm text-slate-600 underline">
            Back
          </Link>
        </div>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Medications &amp; remedies</h2>
          {meds.length === 0 && (
            <p className="text-sm text-slate-500">
              No active medications.{" "}
              <Link href="/medications" className="underline">
                Add one
              </Link>.
            </p>
          )}
          {meds.map((m) => {
            const em = takenMap.get(m.id)
            return (
              <div
                key={m.id}
                className="flex items-center justify-between border-b border-slate-100 py-2 last:border-0"
              >
                <div>
                  <div className="text-sm font-medium">{m.name}</div>
                  <div className="text-xs text-slate-500">
                    {[m.dose, m.frequency].filter(Boolean).join(" · ")}
                  </div>
                </div>
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={!!em?.taken}
                    onCheckedChange={(v) => toggleMed(m, !!v)}
                  />
                  Taken
                </label>
              </div>
            )
          })}
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Suspected triggers</h2>
          {triggerGroups.length === 0 && (
            <p className="text-sm text-slate-500">Loading triggers…</p>
          )}
          {triggerGroups.map(([group, rows]) => (
            <div key={group}>
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-1">
                {group}
              </div>
              <div className="flex flex-wrap gap-2">
                {rows.map((t) => {
                  const active = activeTriggers.has(t.key)
                  return (
                    <Button
                      key={t.key}
                      size="sm"
                      variant={active ? "default" : "outline"}
                      onClick={() => toggleTrigger(t.key, !active)}
                    >
                      {t.label}
                    </Button>
                  )
                })}
              </div>
            </div>
          ))}
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Products used today</h2>
          <div className="space-y-1">
            {products.length === 0 && (
              <p className="text-sm text-slate-500">None logged.</p>
            )}
            {products.map((p) => (
              <div
                key={p.id}
                className="flex items-center justify-between border-b border-slate-100 py-1 last:border-0 text-sm"
              >
                <div>
                  <span className="font-medium">{p.product_name}</span>{" "}
                  <span className="text-xs text-slate-500">
                    · {p.product_type.replace("_", " ")}
                  </span>
                  {p.first_use && (
                    <span className="ml-1 text-xs text-amber-700">
                      (first use)
                    </span>
                  )}
                  {p.adverse_reaction && (
                    <span className="ml-1 text-xs text-red-700">
                      (reaction)
                    </span>
                  )}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => removeProduct(p.id)}
                >
                  Remove
                </Button>
              </div>
            ))}
          </div>
          <div className="border-t border-slate-200 pt-3 space-y-2">
            <div>
              <Label htmlFor="pname">Add product</Label>
              <Input
                id="pname"
                value={newProduct.product_name}
                onChange={(e) =>
                  setNewProduct({ ...newProduct, product_name: e.target.value })
                }
                placeholder="e.g. CeraVe cream"
              />
            </div>
            <div>
              <Label>Type</Label>
              <Select
                value={newProduct.product_type}
                onValueChange={(v) =>
                  setNewProduct({ ...newProduct, product_type: v as MedCategory })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PRODUCT_TYPE_OPTIONS.map((p) => (
                    <SelectItem key={p.value} value={p.value}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={newProduct.first_use}
                onCheckedChange={(v) =>
                  setNewProduct({ ...newProduct, first_use: !!v })
                }
              />
              First time using this
            </label>
            <Textarea
              rows={2}
              value={newProduct.notes}
              onChange={(e) =>
                setNewProduct({ ...newProduct, notes: e.target.value })
              }
              placeholder="Notes (optional)"
            />
            <Button onClick={addProduct}>Add product</Button>
          </div>
        </Card>
      </div>
    </div>
  )
}
