"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { toast } from "sonner"
import { useAuth } from "@/context/AuthContext"
import { apiGet, apiSend } from "@/lib/api/client"
import {
  CURRENT_CONSENT_VERSION,
  CURRENT_SYMPTOM_SCALE_VERSION,
  type AppPreferences,
  type Condition,
  type Profile,
} from "@/lib/types/backend"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"

type Step = 0 | 1 | 2 | 3 | 4

type LocalPrefs = {
  preferred_log_time: string
  reminders_enabled: boolean
}

export default function OnboardingPage() {
  const router = useRouter()
  const { user, loading } = useAuth()
  const [step, setStep] = useState<Step>(0)
  const [disclaimerAck, setDisclaimerAck] = useState(false)
  const [conditions, setConditions] = useState<Condition[]>([])
  const [selectedConditions, setSelectedConditions] = useState<Set<string>>(new Set())
  const [displayName, setDisplayName] = useState("")
  const [allergies, setAllergies] = useState("")
  const [clinicNotes, setClinicNotes] = useState("")
  const [prefs, setPrefs] = useState<LocalPrefs>({
    preferred_log_time: "20:00:00",
    reminders_enabled: false,
  })
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/onboarding")
  }, [user, loading, router])

  useEffect(() => {
    if (!user) return
    apiGet<Condition[]>("/api/conditions").then(setConditions).catch(() => {})
    apiGet<Profile>("/api/profile").then((p) => {
      if (p?.onboarding_completed_at) router.replace("/")
      if (p?.display_name) setDisplayName(p.display_name)
      if (p?.clinic_notes) setClinicNotes(p.clinic_notes)
    }).catch(() => {})
    apiGet<AppPreferences>("/api/app-preferences").then((p) => {
      setPrefs({
        preferred_log_time: p.preferred_log_time ?? "20:00:00",
        reminders_enabled: p.reminders_enabled ?? false,
      })
    }).catch(() => {})
  }, [user, router])

  async function complete() {
    setSaving(true)
    try {
      await apiSend("/api/profile", "PUT", {
        display_name: displayName.trim() || null,
        clinic_notes: clinicNotes.trim() || null,
        onboarding_completed_at: new Date().toISOString(),
        consent_acknowledged_at: new Date().toISOString(),
        consent_version: CURRENT_CONSENT_VERSION,
        symptom_scale_version: CURRENT_SYMPTOM_SCALE_VERSION,
      })
      await apiSend("/api/app-preferences", "PUT", {
        completed_onboarding: true,
        consent_version: CURRENT_CONSENT_VERSION,
        preferred_log_time: prefs.preferred_log_time,
        reminders_enabled: prefs.reminders_enabled,
      })
      for (const conditionId of selectedConditions) {
        await apiSend("/api/user-conditions", "POST", {
          condition_id: conditionId,
          source: "self_reported",
        }).catch(() => {})
      }
      const allergyLines = allergies
        .split(/[\n,]/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
      for (const line of allergyLines) {
        await apiSend("/api/user-allergies", "POST", {
          allergen: line,
        }).catch(() => {})
      }
      toast.success("You're all set.")
      router.replace("/")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not finish setup")
    } finally {
      setSaving(false)
    }
  }

  if (loading || !user) {
    return <div className="p-8 text-center">Loading…</div>
  }

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-xl space-y-4">
        <h1 className="text-2xl font-semibold">Set up SkinTrack+</h1>
        <p className="text-sm text-slate-600">Step {step + 1} of 5</p>

        <Card className="p-5 space-y-4">
          {step === 0 && (
            <div className="space-y-3">
              <h2 className="text-lg font-medium">Important: What this app is not</h2>
              <p className="text-sm text-slate-700">
                SkinTrack+ is a <strong>self-tracking tool</strong>. It is not a
                medical device, it does not diagnose conditions, and it is not a
                substitute for evaluation by a licensed healthcare provider.
              </p>
              <p className="text-sm text-slate-700">
                If you see a concerning change, contact your dermatologist. In an
                emergency, call your local emergency number.
              </p>
              <label className="flex items-start gap-2 text-sm">
                <Checkbox
                  checked={disclaimerAck}
                  onCheckedChange={(v) => setDisclaimerAck(!!v)}
                />
                <span>
                  I understand SkinTrack+ is not a diagnostic tool and will not
                  replace professional medical advice.
                </span>
              </label>
              <p className="text-xs text-slate-500">
                By continuing you agree to the{" "}
                <Link className="underline" href="/legal/terms" target="_blank">
                  Terms
                </Link>{" "}
                and{" "}
                <Link className="underline" href="/legal/privacy" target="_blank">
                  Privacy Policy
                </Link>.
              </p>
            </div>
          )}

          {step === 1 && (
            <div className="space-y-3">
              <h2 className="text-lg font-medium">About you</h2>
              <div>
                <Label htmlFor="displayName">Display name</Label>
                <Input
                  id="displayName"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="What should we call you?"
                />
              </div>
              <div>
                <Label htmlFor="allergies">Allergies or sensitivities</Label>
                <Textarea
                  id="allergies"
                  rows={3}
                  value={allergies}
                  onChange={(e) => setAllergies(e.target.value)}
                  placeholder="One per line — e.g. fragrance, nickel, peanuts"
                />
              </div>
              <div>
                <Label htmlFor="derm">Dermatologist / clinic notes</Label>
                <Textarea
                  id="derm"
                  rows={3}
                  value={clinicNotes}
                  onChange={(e) => setClinicNotes(e.target.value)}
                  placeholder="Clinic name, provider, next visit"
                />
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-3">
              <h2 className="text-lg font-medium">Conditions you track</h2>
              <p className="text-sm text-slate-600">
                Pick any that apply. This is for tracking only — it is not a
                diagnosis.
              </p>
              <div className="space-y-2">
                {conditions.map((c) => {
                  const selected = selectedConditions.has(c.id)
                  return (
                    <label
                      key={c.id}
                      className="flex items-center gap-2 rounded border border-slate-200 p-2 text-sm"
                    >
                      <Checkbox
                        checked={selected}
                        onCheckedChange={(v) => {
                          setSelectedConditions((prev) => {
                            const next = new Set(prev)
                            if (v) next.add(c.id)
                            else next.delete(c.id)
                            return next
                          })
                        }}
                      />
                      <span>{c.display_name}</span>
                    </label>
                  )
                })}
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-3">
              <h2 className="text-lg font-medium">Reminders</h2>
              <p className="text-sm text-slate-600">
                Optional. Reminders fire while the app is open. You can change
                this in Settings.
              </p>
              <label className="flex items-center gap-2 text-sm">
                <Checkbox
                  checked={prefs.reminders_enabled}
                  onCheckedChange={(v) =>
                    setPrefs((p) => ({ ...p, reminders_enabled: !!v }))
                  }
                />
                <span>Daily log reminder</span>
              </label>
              <div>
                <Label htmlFor="logTime">Preferred reminder time</Label>
                <Input
                  id="logTime"
                  type="time"
                  value={prefs.preferred_log_time.slice(0, 5)}
                  onChange={(e) =>
                    setPrefs((p) => ({
                      ...p,
                      preferred_log_time: `${e.target.value}:00`,
                    }))
                  }
                />
              </div>
            </div>
          )}

          {step === 4 && (
            <div className="space-y-3">
              <h2 className="text-lg font-medium">Review</h2>
              <div className="text-sm text-slate-700 space-y-1">
                <div><strong>Name:</strong> {displayName || "(not set)"}</div>
                <div>
                  <strong>Conditions:</strong>{" "}
                  {selectedConditions.size === 0
                    ? "(none)"
                    : conditions
                        .filter((c) => selectedConditions.has(c.id))
                        .map((c) => c.display_name)
                        .join(", ")}
                </div>
                <div>
                  <strong>Reminders:</strong>{" "}
                  {prefs.reminders_enabled
                    ? `daily log at ${prefs.preferred_log_time.slice(0, 5)}`
                    : "off"}
                </div>
              </div>
              <p className="text-xs text-slate-500">
                You can edit everything later in Settings.
              </p>
            </div>
          )}

          <div className="flex justify-between pt-2">
            <Button
              variant="ghost"
              onClick={() => setStep((s) => (Math.max(0, s - 1) as Step))}
              disabled={step === 0}
            >
              Back
            </Button>
            {step < 4 ? (
              <Button
                onClick={() => setStep((s) => (Math.min(4, s + 1) as Step))}
                disabled={step === 0 && !disclaimerAck}
              >
                Next
              </Button>
            ) : (
              <Button onClick={complete} disabled={saving}>
                {saving ? "Saving…" : "Finish setup"}
              </Button>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
