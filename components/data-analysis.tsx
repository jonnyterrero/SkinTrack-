"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, TrendingUp, TrendingDown, Minus } from "lucide-react"
import type { SkinTrackRecord, SymptomTrackRecord } from "@/lib/types"
import { isSymptomRecord } from "@/lib/types"

type Trend = "increasing" | "decreasing" | "stable"

function TrendIcon({ trend }: { trend: Trend }) {
  switch (trend) {
    case "increasing":
      return <TrendingUp className="w-4 h-4 text-red-500" />
    case "decreasing":
      return <TrendingDown className="w-4 h-4 text-green-500" />
    default:
      return <Minus className="w-4 h-4 text-gray-500" />
  }
}

type Props = {
  records: SkinTrackRecord[]
}

export default function DataAnalysis({ records }: Props) {
  const symptomRecords = records.filter(isSymptomRecord)

  const downloadData = () => {
    const dataStr = JSON.stringify(records, null, 2)
    const dataBlob = new Blob([dataStr], { type: "application/json" })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement("a")
    link.href = url
    link.download = "skintrack-data.json"
    link.click()
    URL.revokeObjectURL(url)
  }

  const getAverage = (field: keyof Pick<SymptomTrackRecord, "itch" | "pain" | "stress" | "sleep">) => {
    if (symptomRecords.length === 0) return "0"
    const sum = symptomRecords.reduce((acc, record) => acc + (Number(record[field]) || 0), 0)
    return (sum / symptomRecords.length).toFixed(1)
  }

  const getTrend = (field: keyof Pick<SymptomTrackRecord, "itch" | "pain" | "stress" | "sleep">): Trend => {
    if (symptomRecords.length < 2) return "stable"
    const recent = symptomRecords.slice(-3)
    const older = symptomRecords.slice(-6, -3)

    if (recent.length === 0 || older.length === 0) return "stable"

    const recentAvg = recent.reduce((acc, r) => acc + (Number(r[field]) || 0), 0) / recent.length
    const olderAvg = older.reduce((acc, r) => acc + (Number(r[field]) || 0), 0) / older.length

    const diff = recentAvg - olderAvg
    if (diff > 0.5) return "increasing"
    if (diff < -0.5) return "decreasing"
    return "stable"
  }

  if (records.length === 0) {
    return (
      <Card className="border-dashed border-cyan-200/70 dark:border-cyan-900/50">
        <CardHeader>
          <CardTitle>Data Analysis</CardTitle>
          <CardDescription>No data available for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Start tracking symptoms and capturing images to see your progress here. If analysis ever fails, try a smaller
            image or export a backup from the Data tab.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Overview Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg dark:bg-blue-950/30">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{records.length}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Records</div>
            </div>

            <div className="text-center p-4 bg-green-50 rounded-lg dark:bg-green-950/30">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {records.filter((r) => r.type === "image").length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Images</div>
            </div>

            <div className="rounded-lg bg-cyan-50 p-4 text-center dark:bg-cyan-950/40">
              <div className="text-2xl font-bold text-cyan-700 dark:text-cyan-400">{symptomRecords.length}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Symptom Records</div>
            </div>

            <div className="text-center p-4 bg-orange-50 rounded-lg dark:bg-orange-950/30">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {new Set(symptomRecords.map((r) => r.condition)).size}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Conditions</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {symptomRecords.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Symptom Trends</CardTitle>
            <CardDescription>Average levels and recent trends (based on your log order)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-red-50 rounded-lg dark:bg-red-950/30">
                  <div>
                    <div className="font-semibold">Itch Level</div>
                    <div className="text-2xl font-bold text-red-600 dark:text-red-400">{getAverage("itch")}/10</div>
                  </div>
                  <TrendIcon trend={getTrend("itch")} />
                </div>

                <div className="flex items-center justify-between p-4 bg-orange-50 rounded-lg dark:bg-orange-950/30">
                  <div>
                    <div className="font-semibold">Pain Level</div>
                    <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{getAverage("pain")}/10</div>
                  </div>
                  <TrendIcon trend={getTrend("pain")} />
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg dark:bg-blue-950/30">
                  <div>
                    <div className="font-semibold">Stress Level</div>
                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{getAverage("stress")}/10</div>
                  </div>
                  <TrendIcon trend={getTrend("stress")} />
                </div>

                <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg dark:bg-green-950/30">
                  <div>
                    <div className="font-semibold">Sleep Hours</div>
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">{getAverage("sleep")}h</div>
                  </div>
                  <TrendIcon trend={getTrend("sleep")} />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Export Data</CardTitle>
          <CardDescription>Download raw records JSON (use Data tab for versioned backup with profile)</CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={downloadData} className="w-full">
            <Download className="w-4 h-4 mr-2" />
            Download Data (JSON)
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
