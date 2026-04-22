"use client"

import { useCallback, useEffect, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import { toast } from "sonner"
import { useAuth } from "@/context/AuthContext"
import { apiGet, apiSend } from "@/lib/api/client"
import { BodyMap, type BodyPin } from "@/components/body-map"
import type { LesionLocation } from "@/lib/types/backend"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type LesionRow = { id: string; label: string }

export default function BodyMapPage() {
  const router = useRouter()
  const params = useSearchParams()
  const { user, loading } = useAuth()
  const [lesions, setLesions] = useState<LesionRow[]>([])
  const [selectedLesion, setSelectedLesion] = useState<string | null>(null)
  const [pins, setPins] = useState<BodyPin[]>([])

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/body-map")
  }, [user, loading, router])

  useEffect(() => {
    if (!user) return
    apiGet<LesionRow[]>("/api/lesions")
      .then((list) => {
        setLesions(list)
        const initial = params.get("lesion") ?? list[0]?.id ?? null
        setSelectedLesion(initial)
      })
      .catch(() => {})
  }, [user, params])

  const reload = useCallback(async () => {
    if (!selectedLesion) return
    const rows = await apiGet<LesionLocation[]>(
      `/api/lesion-locations?lesion_id=${selectedLesion}`,
    )
    setPins(
      rows.map((r) => ({
        id: r.id,
        body_view: r.body_view,
        body_region: r.body_region,
        side: r.side,
        loc_x: Number(r.loc_x),
        loc_y: Number(r.loc_y),
      })),
    )
  }, [selectedLesion])

  useEffect(() => {
    reload()
  }, [reload])

  async function add(pin: BodyPin) {
    if (!selectedLesion) {
      toast.error("Pick a lesion first.")
      return
    }
    try {
      await apiSend("/api/lesion-locations", "POST", {
        lesion_id: selectedLesion,
        body_view: pin.body_view,
        body_region: pin.body_region,
        side: pin.side,
        loc_x: Number(pin.loc_x.toFixed(4)),
        loc_y: Number(pin.loc_y.toFixed(4)),
      })
      toast.success(`Pinned ${pin.body_region}`)
      reload()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not save pin")
    }
  }

  async function remove(pin: BodyPin) {
    if (!pin.id) return
    try {
      await apiSend(`/api/lesion-locations/${pin.id}`, "DELETE")
      reload()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not delete pin")
    }
  }

  if (loading || !user) return <div className="p-8 text-center">Loading…</div>

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-xl space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Body map</h1>
          <Link href="/" className="text-sm text-slate-600 underline">
            Back
          </Link>
        </div>

        <Card className="p-4 space-y-3">
          <div>
            <Label>Lesion</Label>
            <Select
              value={selectedLesion ?? ""}
              onValueChange={(v) => setSelectedLesion(v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Pick a lesion" />
              </SelectTrigger>
              <SelectContent>
                {lesions.map((l) => (
                  <SelectItem key={l.id} value={l.id}>
                    {l.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {lesions.length === 0 && (
              <p className="mt-2 text-xs text-amber-700">
                No lesions yet. Create one in the main app first.
              </p>
            )}
          </div>

          <BodyMap pins={pins} onAdd={add} onRemove={remove} />

          <div className="text-xs text-slate-600 space-y-1">
            <div>Pins for this lesion: {pins.length}</div>
            {pins.length > 0 && (
              <ul className="list-disc pl-4 space-y-0.5">
                {pins.map((p, i) => (
                  <li key={p.id ?? `p${i}`}>
                    {p.body_view} · {p.body_region.replace("_", " ")}
                    {p.side ? ` (${p.side})` : ""}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <Button
            variant="outline"
            onClick={() => router.back()}
          >
            Done
          </Button>
        </Card>
      </div>
    </div>
  )
}
