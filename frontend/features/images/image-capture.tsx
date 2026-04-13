"use client"

import { useState, useRef, type ChangeEvent } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Camera, Upload, Save } from "lucide-react"
import { BODY_AREA_DEFINITIONS } from "@/lib/domain/body-areas"
import type { ImageMetadata } from "@/lib/types"

type Props = {
  onImageCaptured: (payload: { type: "image"; filename: string; image: string; metadata?: ImageMetadata }) => void
}

function readImageDimensions(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => reject(new Error("Could not read image dimensions"))
    img.src = dataUrl
  })
}

export default function ImageCapture({ onImageCaptured }: Props) {
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [bodyArea, setBodyArea] = useState<string>("")
  const [note, setNote] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result
        if (typeof result === "string") setCapturedImage(result)
      }
      reader.readAsDataURL(file)
    }
  }

  const saveImage = async () => {
    if (!capturedImage) return
    let width: number | undefined
    let height: number | undefined
    try {
      const dims = await readImageDimensions(capturedImage)
      width = dims.width
      height = dims.height
    } catch {
      /* optional */
    }
    const metadata: ImageMetadata = {
      capturedAt: new Date().toISOString(),
      source: "upload",
      ...(bodyArea ? { bodyArea } : {}),
      ...(note.trim() ? { note: note.trim() } : {}),
      ...(width && height ? { width, height } : {}),
    }
    onImageCaptured({
      image: capturedImage,
      filename: imageFile?.name || `capture_${Date.now()}.jpg`,
      type: "image",
      metadata,
    })
    setCapturedImage(null)
    setImageFile(null)
    setBodyArea("")
    setNote("")
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Image
          </CardTitle>
          <CardDescription>Upload an existing image of your skin condition for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileUpload} className="cursor-pointer" />

            {capturedImage ? (
              <div className="space-y-4">
                <div className="overflow-hidden rounded-lg border">
                  <img
                    src={capturedImage || "/placeholder.svg"}
                    alt="Captured skin condition"
                    className="h-auto max-h-96 w-full object-contain"
                  />
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Body area (optional)</Label>
                    <Select value={bodyArea || "__none__"} onValueChange={(v) => setBodyArea(v === "__none__" ? "" : v)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Link to map" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__none__">Not specified</SelectItem>
                        {BODY_AREA_DEFINITIONS.map((a) => (
                          <SelectItem key={a.id} value={a.id}>
                            {a.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="img-note">Note</Label>
                    <Textarea
                      id="img-note"
                      rows={2}
                      value={note}
                      onChange={(e) => setNote(e.target.value)}
                      placeholder="Lighting, context…"
                      className="resize-none"
                    />
                  </div>
                </div>

                <Button onClick={() => void saveImage()} className="w-full">
                  <Save className="mr-2 h-4 w-4" />
                  Save Image
                </Button>
              </div>
            ) : null}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Camera Capture
          </CardTitle>
          <CardDescription>Take a photo directly with your device camera</CardDescription>
        </CardHeader>
        <CardContent>
          <Input type="file" accept="image/*" capture="environment" onChange={handleFileUpload} className="cursor-pointer" />
          <p className="mt-2 text-sm text-gray-500">This will open your device&apos;s camera app</p>
        </CardContent>
      </Card>
    </div>
  )
}
