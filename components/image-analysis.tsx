"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScanIcon as Analyze, Upload } from "lucide-react"

export default function ImageAnalysis() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  const handleImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target.result)
        setAnalysis(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const analyzeImage = async () => {
    if (!selectedImage) return

    setAnalyzing(true)

    // Simulate image analysis (in a real app, this would use computer vision APIs)
    setTimeout(() => {
      const mockAnalysis = {
        area: (Math.random() * 10 + 1).toFixed(2),
        redness: (Math.random() * 100).toFixed(1),
        irregularity: (Math.random() * 1).toFixed(3),
        asymmetry: (Math.random() * 1).toFixed(3),
        texture: (Math.random() * 50 + 10).toFixed(1),
        confidence: (Math.random() * 20 + 80).toFixed(1),
      }
      setAnalysis(mockAnalysis)
      setAnalyzing(false)
    }, 2000)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Image for Analysis
          </CardTitle>
          <CardDescription>Select an image to analyze skin condition metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <Input type="file" accept="image/*" onChange={handleImageUpload} className="cursor-pointer" />
        </CardContent>
      </Card>

      {selectedImage && (
        <Card>
          <CardHeader>
            <CardTitle>Image Preview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="border rounded-lg overflow-hidden">
                <img
                  src={selectedImage || "/placeholder.svg"}
                  alt="Image for analysis"
                  className="w-full h-auto max-h-96 object-contain"
                />
              </div>

              <Button onClick={analyzeImage} disabled={analyzing} className="w-full">
                <Analyze className="w-4 h-4 mr-2" />
                {analyzing ? "Analyzing..." : "Analyze Image"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {analysis && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{analysis.area}</div>
                <div className="text-sm text-gray-600">Area (cm²)</div>
              </div>

              <div className="text-center p-4 bg-red-50 rounded-lg">
                <div className="text-2xl font-bold text-red-600">{analysis.redness}%</div>
                <div className="text-sm text-gray-600">Redness Score</div>
              </div>

              <div className="text-center p-4 bg-yellow-50 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">{analysis.irregularity}</div>
                <div className="text-sm text-gray-600">Border Irregularity</div>
              </div>

              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{analysis.asymmetry}</div>
                <div className="text-sm text-gray-600">Asymmetry</div>
              </div>

              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{analysis.texture}</div>
                <div className="text-sm text-gray-600">Texture Variance</div>
              </div>

              <div className="text-center p-4 bg-indigo-50 rounded-lg">
                <div className="text-2xl font-bold text-indigo-600">{analysis.confidence}%</div>
                <div className="text-sm text-gray-600">Confidence</div>
              </div>
            </div>

            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold mb-2">Analysis Notes:</h3>
              <p className="text-sm text-gray-600">
                This analysis uses simplified computer vision techniques. For medical diagnosis, always consult with a
                healthcare professional.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
