"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface Profile {
  name: string
  age: string
  gender: string
  skinType: string
  conditions: string
  medications: string
  allergies: string
  notes: string
}

export default function ProfileManager() {
  const [profile, setProfile] = useState<Profile>({
    name: "",
    age: "",
    gender: "",
    skinType: "",
    conditions: "",
    medications: "",
    allergies: "",
    notes: "",
  })

  useEffect(() => {
    const savedProfile = localStorage.getItem("skintrack-profile")
    if (savedProfile) {
      setProfile(JSON.parse(savedProfile))
    }
  }, [])

  const handleSave = () => {
    localStorage.setItem("skintrack-profile", JSON.stringify(profile))
    alert("Profile saved successfully!")
  }

  const handleInputChange = (field: keyof Profile, value: string) => {
    setProfile((prev) => ({ ...prev, [field]: value }))
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-2">
          👤 Profile Management
        </h2>
        <p className="text-foreground/70">Manage your personal information and medical history</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card className="glass-card border-white/20">
          <CardHeader>
            <CardTitle className="text-lg">Personal Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="name">Full Name</Label>
              <Input
                id="name"
                value={profile.name}
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
                value={profile.age}
                onChange={(e) => handleInputChange("age", e.target.value)}
                className="glass-input"
                placeholder="Enter your age"
              />
            </div>
            <div>
              <Label htmlFor="gender">Gender</Label>
              <Select value={profile.gender} onValueChange={(value) => handleInputChange("gender", value)}>
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
              <Select value={profile.skinType} onValueChange={(value) => handleInputChange("skinType", value)}>
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

        <Card className="glass-card border-white/20">
          <CardHeader>
            <CardTitle className="text-lg">Medical Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="conditions">Skin Conditions</Label>
              <Textarea
                id="conditions"
                value={profile.conditions}
                onChange={(e) => handleInputChange("conditions", e.target.value)}
                className="glass-input"
                placeholder="List any known skin conditions..."
                rows={3}
              />
            </div>
            <div>
              <Label htmlFor="medications">Current Medications</Label>
              <Textarea
                id="medications"
                value={profile.medications}
                onChange={(e) => handleInputChange("medications", e.target.value)}
                className="glass-input"
                placeholder="List current medications..."
                rows={3}
              />
            </div>
            <div>
              <Label htmlFor="allergies">Known Allergies</Label>
              <Textarea
                id="allergies"
                value={profile.allergies}
                onChange={(e) => handleInputChange("allergies", e.target.value)}
                className="glass-input"
                placeholder="List any known allergies..."
                rows={3}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card border-white/20">
        <CardHeader>
          <CardTitle className="text-lg">Additional Notes</CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea
            value={profile.notes}
            onChange={(e) => handleInputChange("notes", e.target.value)}
            className="glass-input"
            placeholder="Any additional notes about your skin health journey..."
            rows={4}
          />
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button
          onClick={handleSave}
          className="glass-button bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white border-0 px-8"
        >
          💾 Save Profile
        </Button>
      </div>
    </div>
  )
}
