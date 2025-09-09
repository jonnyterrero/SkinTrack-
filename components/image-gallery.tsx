"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function ImageGallery({ records }) {
  const imageRecords = records.filter((r) => r.type === "image" && r.image)

  if (imageRecords.length === 0) {
    return null
  }

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle>📸 Recent Images ({imageRecords.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {imageRecords.slice(0, 6).map((record) => (
            <div key={record.id} className="border rounded-lg overflow-hidden">
              <img
                src={record.image || "/placeholder.svg"}
                alt={record.filename}
                className="w-full h-48 object-cover"
              />
              <div className="p-3">
                <div className="text-sm font-medium truncate">{record.filename}</div>
                <div className="text-xs text-gray-500">{new Date(record.timestamp).toLocaleDateString()}</div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
