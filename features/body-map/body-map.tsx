"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BODY_AREA_DEFINITIONS } from "@/lib/domain/body-areas"
import type { SkinTrackRecord } from "@/lib/types"

interface BodyMapProps {
  records: SkinTrackRecord[]
}

export default function BodyMap({ records }: BodyMapProps) {
  const [selectedArea, setSelectedArea] = useState<string | null>(null)

  const getRecordsForArea = (areaId: string) => {
    return records.filter((record) => {
      if (record.type === "symptom" && record.bodyArea === areaId) return true
      if (record.type === "image" && record.metadata?.bodyArea === areaId) return true
      return false
    })
  }

  const getAreaColor = (areaId: string) => {
    const areaRecords = getRecordsForArea(areaId)
    if (areaRecords.length === 0) return "bg-gray-400/30"

    const recentRecord = areaRecords[0]
    if (recentRecord.type === "symptom") {
      if (recentRecord.severity === "high") return "bg-red-500/60"
      if (recentRecord.severity === "medium") return "bg-yellow-500/60"
    }
    return "bg-green-500/60"
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="mb-2 bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-3xl font-bold text-transparent">
          Body map
        </h2>
        <p className="text-foreground/70">Visual representation of tracked areas on your body</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="text-lg">Interactive Body Map</CardTitle>
            <CardDescription>Click on body areas to view records</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative mx-auto w-full max-w-md">
              <svg viewBox="0 0 100 100" className="glass h-96 w-full rounded-lg border">
                <ellipse cx="50" cy="12" rx="8" ry="10" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="45" y="22" width="10" height="15" rx="2" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="42" y="37" width="16" height="25" rx="3" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="35" y="30" width="8" height="20" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="57" y="30" width="8" height="20" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="44" y="62" width="12" height="20" rx="2" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="40" y="82" width="8" height="15" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <rect x="52" y="82" width="8" height="15" rx="4" fill="none" stroke="currentColor" strokeWidth="0.5" />

                {BODY_AREA_DEFINITIONS.map((area) => (
                  <circle
                    key={area.id}
                    cx={area.x}
                    cy={area.y}
                    r="3"
                    className={`cursor-pointer transition-all hover:r-4 ${getAreaColor(area.id)} ${
                      selectedArea === area.id ? "stroke-cyan-600 stroke-2" : "stroke-slate-300 stroke-1 dark:stroke-slate-600"
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

        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="text-lg">
              {selectedArea ? `Records for ${BODY_AREA_DEFINITIONS.find((a) => a.id === selectedArea)?.name}` : "Select an Area"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedArea ? (
              <div className="space-y-4">
                {getRecordsForArea(selectedArea).length > 0 ? (
                  getRecordsForArea(selectedArea).map((record, index) => (
                    <div key={index} className="glass rounded-lg border border-white/10 p-4">
                      <div className="mb-2 flex items-start justify-between">
                        <span className="font-medium">{record.type === "image" ? "Image" : "Symptom"}</span>
                        <Badge
                          variant={
                            "severity" in record && record.severity === "high"
                              ? "destructive"
                              : "severity" in record && record.severity === "medium"
                                ? "secondary"
                                : "default"
                          }
                        >
                          {"severity" in record ? record.severity ?? "—" : "—"}
                        </Badge>
                      </div>
                      <p className="mb-2 text-sm text-foreground/70">{new Date(record.timestamp).toLocaleDateString()}</p>
                      {"notes" in record && record.notes ? <p className="text-sm">{record.notes}</p> : null}
                    </div>
                  ))
                ) : (
                  <div className="py-8 text-center text-foreground/60">
                    <p>No records found for this area</p>
                    <p className="mt-2 text-sm">Log a symptom with a body area in Insights, or capture an image with a location in Scan.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-8 text-center text-foreground/60">
                <p>Click on a body area to view records</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
