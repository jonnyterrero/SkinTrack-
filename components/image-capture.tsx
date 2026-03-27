"use client"

import { useState, useRef, type ChangeEvent } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Camera, Upload, Save } from "lucide-react"

type Props = {
  onImageCaptured: (payload: { type: "image"; filename: string; image: string }) => void
}

export default function ImageCapture({ onImageCaptured }: Props) {
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
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

  const saveImage = () => {
    if (capturedImage) {
      onImageCaptured({
        image: capturedImage,
        filename: imageFile?.name || `capture_${Date.now()}.jpg`,
        type: "image",
      })
      setCapturedImage(null)
      setImageFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Image
          </CardTitle>
          <CardDescription>Upload an existing image of your skin condition for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="cursor-pointer"
            />

            {capturedImage && (
              <div className="space-y-4">
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={capturedImage || "/placeholder.svg"}
                    alt="Captured skin condition"
                    className="w-full h-auto max-h-96 object-contain"
                  />
                </div>

                <Button onClick={saveImage} className="w-full">
                  <Save className="w-4 h-4 mr-2" />
                  Save Image
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="w-5 h-5" />
            Camera Capture
          </CardTitle>
          <CardDescription>Take a photo directly with your device camera</CardDescription>
        </CardHeader>
        <CardContent>
          <Input type="file" accept="image/*" capture="environment" onChange={handleFileUpload} className="cursor-pointer" />
          <p className="text-sm text-gray-500 mt-2">This will open your device&apos;s camera app</p>
        </CardContent>
      </Card>
    </div>
  )
}
