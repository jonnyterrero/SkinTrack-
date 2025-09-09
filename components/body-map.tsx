"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface BodyMapProps {
  records: any[]
}

export default function BodyMap({ records }: BodyMapProps) {
  const [selectedArea, setSelectedArea] = useState<string | null>(null)

  const bodyAreas = [
    { id: "head", name: "Head & Face", x: 50, y: 15 },
    { id: "neck", name: "Neck", x: 50, y: 25 },
    { id: "chest", name: "Chest", x: 50, y: 35 },
    { id: "left-arm", name: "Left Arm", x: 25, y: 40 },
    { id: "right-arm", name: "Right Arm", x: 75, y: 40 },
    { id: "abdomen", name: "Abdomen", x: 50, y: 50 },
    { id: "left-hand", name: "Left Hand", x: 15, y: 55 },
    { id: "right-hand", name: "Right Hand", x: 85, y: 55 },
    { id: "pelvis", name: "Pelvis", x: 50, y: 60 },
    { id: "left-thigh", name: "Left Thigh", x: 40, y: 70 },
    { id: "right-thigh", name: "Right Thigh", x: 60, y: 70 },
    { id: "left-knee", name: "Left Knee", x: 40, y: 80 },
    { id: "right-knee", name: "Right Knee", x: 60, y: 80 },
    { id: "left-foot", name: "Left Foot", x: 40, y: 95 },
    { id: "right-foot", name: "Right Foot", x: 60, y: 95 },
  ]

  const getRecordsForArea = (areaId: string) => {
    return records.filter((record) => record.bodyArea === areaId)
  }

  const getAreaColor = (areaId: string) => {
    const areaRecords = getRecordsForArea(areaId)
    if (areaRecords.length === 0) return "bg-gray-400/30"

    const recentRecord = areaRecords[0]
    if (recentRecord.severity === "high") return "bg-red-500/60"
    if (recentRecord.severity === "medium") return "bg-yellow-500/60"
    return "bg-green-500/60"
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-2">
          🗺️ Body Map
        </h2>
        <p className="text-foreground/70">Visual representation of tracked areas on your body</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="glass-card border-white/20">
          <CardHeader>
            <CardTitle className="text-lg">Interactive Body Map</CardTitle>
            <CardDescription>Click on body areas to view records</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative w-full max-w-md mx-auto">
              <svg viewBox="0 0 100 100" className="w-full h-96 border rounded-lg glass">
                {/* Simple body outline */}
                <ellipse cx="50" cy="12" rx="8" ry="10" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="45" y="22" width="10" height="15" rx="2" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="42" y="37" width="16" height="25" rx="3" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="35" y="30" width="8" height="20" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="57" y="30" width="8" height="20" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="44" y="62" width="12" height="20" rx="2" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="40" y="82" width="8" height="15" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="52" y="82" width="8" height="15" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />

                {/* Interactive areas */}
                {bodyAreas.map((area) => (
                  <circle
                    key={area.id}
                    cx={area.x}
                    cy={area.y}
                    r="3"
                    className={`cursor-pointer transition-all hover:r-4 ${getAreaColor(area.id)} ${
                      selectedArea === area.id ? "stroke-purple-500 stroke-2" : "stroke-white/50 stroke-1"
                    }`}
                    onClick={() => setSelectedArea(area.id)}
                  />
                ))}
              </svg>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <Badge variant="secondary" className="bg-green-500/20 text-green-700">
                ● Low Severity
              </Badge>
              <Badge variant="secondary" className="bg-yellow-500/20 text-yellow-700">
                ● Medium Severity
              </Badge>
              <Badge variant="secondary" className="bg-red-500/20 text-red-700">
                ● High Severity
              </Badge>
              <Badge variant="secondary" className="bg-gray-400/20 text-gray-700">
                ● No Records
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-white/20">
          <CardHeader>
            <CardTitle className="text-lg">
              {selectedArea ? `Records for ${bodyAreas.find((a) => a.id === selectedArea)?.name}` : "Select an Area"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedArea ? (
              <div className="space-y-4">
                {getRecordsForArea(selectedArea).length > 0 ? (
                  getRecordsForArea(selectedArea).map((record, index) => (
                    <div key={index} className="p-4 glass rounded-lg border border-white/10">
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-medium">{record.type === "image" ? "📸 Image" : "📝 Symptom"}</span>
                        <Badge
                          variant={
                            record.severity === "high"
                              ? "destructive"
                              : record.severity === "medium"
                                ? "secondary"
                                : "default"
                          }
                        >
                          {record.severity}
                        </Badge>
                      </div>
                      <p className="text-sm text-foreground/70 mb-2">
                        {new Date(record.timestamp).toLocaleDateString()}
                      </p>
                      {record.notes && <p className="text-sm">{record.notes}</p>}
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-foreground/60">
                    <p>No records found for this area</p>
                    <p className="text-sm mt-2">Start tracking by capturing images or adding symptom records</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-foreground/60">
                <p>Click on a body area to view records</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
