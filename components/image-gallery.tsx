"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ImageTrackRecord, SkinTrackRecord } from "@/lib/types"
import { isImageRecord } from "@/lib/types"

type Props = {
  records: SkinTrackRecord[]
}

export default function ImageGallery({ records }: Props) {
  const imageRecords: ImageTrackRecord[] = records.filter(
    (r): r is ImageTrackRecord => isImageRecord(r) && Boolean(r.image || r.imageRef),
  )

  if (imageRecords.length === 0) {
    return (
      <Card className="mt-2 border-dashed border-cyan-200/60 dark:border-cyan-900/50">
        <CardHeader>
          <CardTitle className="text-base text-muted-foreground">Recent images</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No saved photos yet. Use <strong>Scan</strong> to upload or capture an image, then tap <strong>Save</strong>.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle>Recent Images ({imageRecords.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {imageRecords.slice(0, 6).map((record) => (
            <div key={record.id} className="border rounded-lg overflow-hidden">
              {record.image ? (
                <img
                  src={record.image}
                  alt={record.filename}
                  className="w-full h-48 object-cover"
                />
              ) : (
                <div className="flex h-48 items-center justify-center bg-muted text-center text-xs text-muted-foreground px-2">
                  Image file missing (IndexedDB was cleared). Filename: {record.filename}
                </div>
              )}
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
