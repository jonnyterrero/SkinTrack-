"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { toast } from "sonner"
import { useAuth } from "@/context/AuthContext"
import { apiGet, apiSend } from "@/lib/api/client"
import type {
  AppPreferences,
  Profile,
  UserAllergy,
} from "@/lib/types/backend"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"

export default function SettingsPage() {
  const router = useRouter()
  const { user, loading, signOut } = useAuth()
  const [displayName, setDisplayName] = useState("")
  const [clinicNotes, setClinicNotes] = useState("")
  const [prefs, setPrefs] = useState<Partial<AppPreferences>>({})
  const [allergies, setAllergies] = useState<UserAllergy[]>([])
  const [newAllergen, setNewAllergen] = useState("")
  const [saving, setSaving] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState("")
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/settings")
  }, [user, loading, router])

  useEffect(() => {
    if (!user) return
    apiGet<Profile>("/api/profile")
      .then((p) => {
        setDisplayName(p?.display_name ?? "")
        setClinicNotes(p?.clinic_notes ?? "")
      })
      .catch(() => {})
    apiGet<AppPreferences>("/api/app-preferences")
      .then((p) => setPrefs(p))
      .catch(() => {})
    apiGet<UserAllergy[]>("/api/user-allergies")
      .then(setAllergies)
      .catch(() => {})
  }, [user])

  async function save() {
    setSaving(true)
    try {
      await apiSend("/api/profile", "PUT", {
        display_name: displayName.trim() || null,
        clinic_notes: clinicNotes.trim() || null,
      })
      await apiSend("/api/app-preferences", "PUT", {
        preferred_log_time: prefs.preferred_log_time,
        reminders_enabled: prefs.reminders_enabled,
        quiet_hours_start: prefs.quiet_hours_start,
        quiet_hours_end: prefs.quiet_hours_end,
        units: prefs.units,
        theme: prefs.theme,
      })
      toast.success("Saved.")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Save failed")
    } finally {
      setSaving(false)
    }
  }

  async function addAllergen() {
    const value = newAllergen.trim()
    if (!value) return
    try {
      const created = await apiSend<UserAllergy>("/api/user-allergies", "POST", {
        allergen: value,
      })
      setAllergies((prev) => [...prev, created])
      setNewAllergen("")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not add allergy")
    }
  }

  async function removeAllergen(id: string) {
    try {
      await apiSend(`/api/user-allergies/${id}`, "DELETE")
      setAllergies((prev) => prev.filter((a) => a.id !== id))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not remove allergy")
    }
  }

  async function exportMyData() {
    try {
      const data = await apiGet<unknown>("/api/account")
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `skintrack-export-${new Date().toISOString().slice(0, 10)}.json`
      a.click()
      URL.revokeObjectURL(url)
      toast.success("Export downloaded.")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Export failed")
    }
  }

  async function deleteAccount() {
    if (deleteConfirm !== "DELETE") {
      toast.error("Type DELETE to confirm.")
      return
    }
    setDeleting(true)
    try {
      await apiSend("/api/account", "DELETE")
      await signOut()
      toast.success("Account deleted.")
      router.replace("/")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Deletion failed")
    } finally {
      setDeleting(false)
    }
  }

  if (loading || !user) return <div className="p-8 text-center">Loading…</div>

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-2xl space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Settings</h1>
          <Link href="/" className="text-sm text-slate-600 underline">
            Back to app
          </Link>
        </div>

        <Card className="p-5 space-y-4">
          <h2 className="text-lg font-medium">Profile</h2>
          <div>
            <Label htmlFor="name">Display name</Label>
            <Input
              id="name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </div>
          <div>
            <Label htmlFor="email">Email</Label>
            <Input id="email" value={user.email ?? ""} disabled />
          </div>
          <div>
            <Label htmlFor="derm">Dermatologist / clinic notes</Label>
            <Textarea
              id="derm"
              rows={3}
              value={clinicNotes}
              onChange={(e) => setClinicNotes(e.target.value)}
            />
          </div>
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Allergies</h2>
          <div className="space-y-1">
            {allergies.length === 0 && (
              <p className="text-sm text-slate-500">No allergies recorded.</p>
            )}
            {allergies.map((a) => (
              <div
                key={a.id}
                className="flex items-center justify-between rounded border border-slate-200 p-2 text-sm"
              >
                <span>{a.allergen}</span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => removeAllergen(a.id)}
                >
                  Remove
                </Button>
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <Input
              value={newAllergen}
              onChange={(e) => setNewAllergen(e.target.value)}
              placeholder="Add allergen (e.g. fragrance)"
            />
            <Button onClick={addAllergen}>Add</Button>
          </div>
        </Card>

        <Card className="p-5 space-y-4">
          <h2 className="text-lg font-medium">Reminders</h2>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={!!prefs.reminders_enabled}
              onCheckedChange={(v) =>
                setPrefs((p) => ({ ...p, reminders_enabled: !!v }))
              }
            />
            <span>Daily log reminder (fires while app is open)</span>
          </label>
          <div>
            <Label htmlFor="logtime">Reminder time</Label>
            <Input
              id="logtime"
              type="time"
              value={(prefs.preferred_log_time ?? "20:00:00").slice(0, 5)}
              onChange={(e) =>
                setPrefs((p) => ({
                  ...p,
                  preferred_log_time: `${e.target.value}:00`,
                }))
              }
            />
          </div>
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Your data</h2>
          <p className="text-sm text-slate-600">
            Download everything on file or open the dedicated export center.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button onClick={exportMyData}>Export my data (JSON)</Button>
            <Link href="/export">
              <Button variant="outline">Open export center</Button>
            </Link>
            <Link href="/legal/privacy">
              <Button variant="ghost">Privacy policy</Button>
            </Link>
            <Link href="/support">
              <Button variant="ghost">Contact support</Button>
            </Link>
          </div>
        </Card>

        <div className="flex gap-2">
          <Button onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save changes"}
          </Button>
          <Button variant="outline" onClick={signOut}>
            Sign out
          </Button>
        </div>

        <Card className="p-5 space-y-3 border-red-200">
          <h2 className="text-lg font-medium text-red-700">Danger zone</h2>
          <p className="text-sm text-slate-600">
            Deleting your account permanently removes all records, images,
            medications, and preferences. This cannot be undone.
          </p>
          <div>
            <Label htmlFor="confirm">Type DELETE to confirm</Label>
            <Input
              id="confirm"
              value={deleteConfirm}
              onChange={(e) => setDeleteConfirm(e.target.value)}
              placeholder="DELETE"
            />
          </div>
          <Button
            variant="destructive"
            onClick={deleteAccount}
            disabled={deleting || deleteConfirm !== "DELETE"}
          >
            {deleting ? "Deleting…" : "Delete my account"}
          </Button>
        </Card>
      </div>
    </div>
  )
}
