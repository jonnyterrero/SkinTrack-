"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, TrendingUp, TrendingDown, Minus } from "lucide-react"

export default function DataAnalysis({ records }) {
  const symptomRecords = records.filter((r) => r.type === "symptom")

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

  const getAverage = (field) => {
    if (symptomRecords.length === 0) return 0
    const sum = symptomRecords.reduce((acc, record) => acc + (record[field] || 0), 0)
    return (sum / symptomRecords.length).toFixed(1)
  }

  const getTrend = (field) => {
    if (symptomRecords.length < 2) return "stable"
    const recent = symptomRecords.slice(-3)
    const older = symptomRecords.slice(-6, -3)

    if (recent.length === 0 || older.length === 0) return "stable"

    const recentAvg = recent.reduce((acc, r) => acc + (r[field] || 0), 0) / recent.length
    const olderAvg = older.reduce((acc, r) => acc + (r[field] || 0), 0) / older.length

    const diff = recentAvg - olderAvg
    if (diff > 0.5) return "increasing"
    if (diff < -0.5) return "decreasing"
    return "stable"
  }

  const TrendIcon = ({ trend }) => {
    switch (trend) {
      case "increasing":
        return <TrendingUp className="w-4 h-4 text-red-500" />
      case "decreasing":
        return <TrendingDown className="w-4 h-4 text-green-500" />
      default:
        return <Minus className="w-4 h-4 text-gray-500" />
    }
  }

  if (records.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>📈 Data Analysis</CardTitle>
          <CardDescription>No data available for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-500">Start tracking symptoms and capturing images to see your progress here.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>📊 Overview Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{records.length}</div>
              <div className="text-sm text-gray-600">Total Records</div>
            </div>

            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {records.filter((r) => r.type === "image").length}
              </div>
              <div className="text-sm text-gray-600">Images</div>
            </div>

            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{symptomRecords.length}</div>
              <div className="text-sm text-gray-600">Symptom Records</div>
            </div>

            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {new Set(symptomRecords.map((r) => r.condition)).size}
              </div>
              <div className="text-sm text-gray-600">Conditions</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {symptomRecords.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>📈 Symptom Trends</CardTitle>
            <CardDescription>Average levels and recent trends</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-red-50 rounded-lg">
                  <div>
                    <div className="font-semibold">Itch Level</div>
                    <div className="text-2xl font-bold text-red-600">{getAverage("itch")}/10</div>
                  </div>
                  <TrendIcon trend={getTrend("itch")} />
                </div>

                <div className="flex items-center justify-between p-4 bg-orange-50 rounded-lg">
                  <div>
                    <div className="font-semibold">Pain Level</div>
                    <div className="text-2xl font-bold text-orange-600">{getAverage("pain")}/10</div>
                  </div>
                  <TrendIcon trend={getTrend("pain")} />
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                  <div>
                    <div className="font-semibold">Stress Level</div>
                    <div className="text-2xl font-bold text-blue-600">{getAverage("stress")}/10</div>
                  </div>
                  <TrendIcon trend={getTrend("stress")} />
                </div>

                <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
                  <div>
                    <div className="font-semibold">Sleep Hours</div>
                    <div className="text-2xl font-bold text-green-600">{getAverage("sleep")}h</div>
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
          <CardTitle>📤 Export Data</CardTitle>
          <CardDescription>Download your tracking data</CardDescription>
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
