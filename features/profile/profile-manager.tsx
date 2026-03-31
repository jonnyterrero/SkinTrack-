"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useSkinTrack } from "@/components/skintrack-provider"
import { toast } from "sonner"
import type { UserProfile } from "@/lib/types"
import DailyMedCheckoff from "@/features/profile/daily-med-checkoff"
import MedicationCatalog from "@/features/profile/medication-catalog"

export default function ProfileManager() {
  const { profile, setProfile, loading } = useSkinTrack()
  const [draft, setDraft] = useState<UserProfile>(profile)

  useEffect(() => {
    setDraft(profile)
  }, [profile])

  const handleInputChange = (field: keyof UserProfile, value: string) => {
    setDraft((prev) => ({ ...prev, [field]: value }))
  }

  const handleSave = () => {
    setProfile(draft)
    toast.success("Profile saved")
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="mb-2 bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-3xl font-bold text-transparent">
          Profile Management
        </h2>
        <p className="text-foreground/70">
          Stored locally in your browser{loading ? " (loading…)" : ""}. Use export in the Data tab for backups. Structured
          medications below are included in versioned exports.
        </p>
      </div>

      <MedicationCatalog />
      <DailyMedCheckoff />

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="text-lg">Personal Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="name">Full Name</Label>
              <Input
                id="name"
                value={draft.name}
                onChange={(e) => handleInputChange("name", e.target.value)}
                className="glass-input"
                placeholder="Enter your full name"
              />
            </div>
            <div>
              <Label htmlFor="age">Age</Label>
              <Input
                id="age"
                type="number"
                value={draft.age}
                onChange={(e) => handleInputChange("age", e.target.value)}
                className="glass-input"
                placeholder="Enter your age"
              />
            </div>
            <div>
              <Label htmlFor="gender">Gender</Label>
              <Select value={draft.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                <SelectTrigger className="glass-input">
                  <SelectValue placeholder="Select gender" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="male">Male</SelectItem>
                  <SelectItem value="female">Female</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                  <SelectItem value="prefer-not-to-say">Prefer not to say</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="skinType">Skin Type</Label>
              <Select value={draft.skinType} onValueChange={(value) => handleInputChange("skinType", value)}>
                <SelectTrigger className="glass-input">
                  <SelectValue placeholder="Select skin type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="type1">Type I - Very Fair</SelectItem>
                  <SelectItem value="type2">Type II - Fair</SelectItem>
                  <SelectItem value="type3">Type III - Medium</SelectItem>
                  <SelectItem value="type4">Type IV - Olive</SelectItem>
                  <SelectItem value="type5">Type V - Brown</SelectItem>
                  <SelectItem value="type6">Type VI - Dark Brown/Black</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="text-lg">Medical Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="conditions">Skin Conditions</Label>
              <Textarea
                id="conditions"
                value={draft.conditions}
                onChange={(e) => handleInputChange("conditions", e.target.value)}
                className="glass-input"
                placeholder="List any known skin conditions..."
                rows={3}
              />
            </div>
            <div>
              <Label htmlFor="medications">Current Medications (free text)</Label>
              <Textarea
                id="medications"
                value={draft.medications}
                onChange={(e) => handleInputChange("medications", e.target.value)}
                className="glass-input"
                placeholder="Summary for your records; use the catalog above for structured tracking."
                rows={3}
              />
            </div>
            <div>
              <Label htmlFor="allergies">Known Allergies</Label>
              <Textarea
                id="allergies"
                value={draft.allergies}
                onChange={(e) => handleInputChange("allergies", e.target.value)}
                className="glass-input"
                placeholder="List any known allergies..."
                rows={3}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Additional Notes</CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea
            value={draft.notes}
            onChange={(e) => handleInputChange("notes", e.target.value)}
            className="glass-input"
            placeholder="Any additional notes about your skin health journey..."
            rows={4}
          />
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} className="glass-button rounded-lg border-0 px-8 text-white">
          Save Profile
        </Button>
      </div>
    </div>
  )
}
