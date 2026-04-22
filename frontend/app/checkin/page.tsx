"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
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
  ProductType,
  TriggerKey,
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

const TRIGGERS: { key: TriggerKey; label: string; group: string }[] = [
  { key: "food", label: "Food (general)", group: "Diet" },
  { key: "dairy", label: "Dairy", group: "Diet" },
  { key: "gluten", label: "Gluten", group: "Diet" },
  { key: "sugar", label: "Sugar", group: "Diet" },
  { key: "alcohol", label: "Alcohol", group: "Diet" },
  { key: "caffeine", label: "Caffeine", group: "Diet" },
  { key: "spicy", label: "Spicy food", group: "Diet" },

  { key: "stress", label: "Stress", group: "Lifestyle" },
  { key: "poor_sleep", label: "Poor sleep", group: "Lifestyle" },
  { key: "menstrual_cycle", label: "Menstrual cycle", group: "Lifestyle" },
  { key: "illness", label: "Illness", group: "Lifestyle" },

  { key: "heat", label: "Heat", group: "Environment" },
  { key: "cold", label: "Cold", group: "Environment" },
  { key: "humidity", label: "Humidity", group: "Environment" },
  { key: "dry_air", label: "Dry air", group: "Environment" },
  { key: "sun", label: "Sun exposure", group: "Environment" },
  { key: "sweat", label: "Sweat", group: "Environment" },

  { key: "detergent", label: "Detergent", group: "Contact" },
  { key: "fragrance", label: "Fragrance", group: "Contact" },
  { key: "new_skincare", label: "New skincare product", group: "Contact" },
  { key: "pet_dander", label: "Pet dander", group: "Contact" },
  { key: "dust", label: "Dust", group: "Contact" },
  { key: "pollen", label: "Pollen", group: "Contact" },
  { key: "new_clothing", label: "New clothing", group: "Contact" },
  { key: "friction", label: "Friction / rubbing", group: "Contact" },
  { key: "other", label: "Other", group: "Contact" },
]

const PRODUCT_TYPES: { value: ProductType; label: string }[] = [
  { value: "moisturizer", label: "Moisturizer" },
  { value: "cleanser", label: "Cleanser" },
  { value: "sunscreen", label: "Sunscreen" },
  { value: "prescription_topical", label: "Prescription topical" },
  { value: "otc_topical", label: "OTC topical" },
  { value: "serum", label: "Serum" },
  { value: "makeup", label: "Makeup" },
  { value: "diet_change", label: "Diet change" },
  { value: "home_remedy", label: "Home remedy" },
  { value: "environmental_change", label: "Environmental change" },
  { value: "avoidance", label: "Avoidance / elimination" },
  { value: "other", label: "Other" },
]

export default function CheckInPage() {
  const router = useRouter()
  const params = useSearchParams()
  const eventId = params.get("event")
  const { user, loading } = useAuth()

  const [meds, setMeds] = useState<MedCatalogItem[]>([])
  const [eventMeds, setEventMeds] = useState<EventMedication[]>([])
  const [triggers, setTriggers] = useState<EventTrigger[]>([])
  const [products, setProducts] = useState<EventProduct[]>([])

  const [newProduct, setNewProduct] = useState({
    product_name: "",
    product_type: "moisturizer" as ProductType,
    first_use: false,
    notes: "",
  })

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/checkin")
  }, [user, loading, router])

  const loadAll = useCallback(async () => {
    if (!eventId) return
    try {
      const [medsList, emList, trigList, prodList] = await Promise.all([
        apiGet<MedCatalogItem[]>("/api/medications?active=true"),
        apiGet<EventMedication[]>(`/api/event-medications?skin_event_id=${eventId}`),
        apiGet<EventTrigger[]>(`/api/event-triggers?skin_event_id=${eventId}`),
        apiGet<EventProduct[]>(`/api/event-products?skin_event_id=${eventId}`),
      ])
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

  async function toggleTrigger(key: TriggerKey, active: boolean) {
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

  const triggerGroups = Array.from(new Set(TRIGGERS.map((t) => t.group)))

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
          {triggerGroups.map((group) => (
            <div key={group}>
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-1">
                {group}
              </div>
              <div className="flex flex-wrap gap-2">
                {TRIGGERS.filter((t) => t.group === group).map((t) => {
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
                  setNewProduct({ ...newProduct, product_type: v as ProductType })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PRODUCT_TYPES.map((p) => (
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
