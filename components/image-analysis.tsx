"use client"

import { useState, type ChangeEvent } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScanIcon as Analyze, Upload } from "lucide-react"

type AnalysisResult = {
  area: string
  redness: string
  irregularity: string
  asymmetry: string
  texture: string
  confidence: string
}

export default function ImageAnalysis() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)

  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result
        if (typeof result === "string") {
          setSelectedImage(result)
          setAnalysis(null)
          setAnalysisError(null)
        }
      }
      reader.onerror = () => {
        setAnalysisError("Could not read that file. Try a smaller image or a different format.")
      }
      reader.readAsDataURL(file)
    }
  }

  const analyzeImage = () => {
    if (!selectedImage) return

    setAnalyzing(true)
    setAnalysisError(null)

    window.setTimeout(() => {
      try {
        const mockAnalysis: AnalysisResult = {
          area: (Math.random() * 10 + 1).toFixed(2),
          redness: (Math.random() * 100).toFixed(1),
          irregularity: (Math.random() * 1).toFixed(3),
          asymmetry: (Math.random() * 1).toFixed(3),
          texture: (Math.random() * 50 + 10).toFixed(1),
          confidence: (Math.random() * 20 + 80).toFixed(1),
        }
        setAnalysis(mockAnalysis)
      } catch {
        setAnalysisError("Analysis step failed. Try again or use a smaller image.")
      } finally {
        setAnalyzing(false)
      }
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
          <CardDescription>
            Demo metrics only in this PWA build. The Python/Streamlit tool can compute real lesion metrics when that
            pipeline is connected.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Input type="file" accept="image/*" onChange={handleImageUpload} className="cursor-pointer" />
        </CardContent>
      </Card>

      {analysisError && (
        <div
          role="alert"
          className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
        >
          {analysisError}
        </div>
      )}

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
            <CardTitle>Analysis Results (demo)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg dark:bg-blue-950/30">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{analysis.area}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Area (cm²)</div>
              </div>

              <div className="text-center p-4 bg-red-50 rounded-lg dark:bg-red-950/30">
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">{analysis.redness}%</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Redness Score</div>
              </div>

              <div className="text-center p-4 bg-yellow-50 rounded-lg dark:bg-yellow-950/30">
                <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{analysis.irregularity}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Border Irregularity</div>
              </div>

              <div className="rounded-lg bg-cyan-50 p-4 text-center dark:bg-cyan-950/40">
                <div className="text-2xl font-bold text-cyan-700 dark:text-cyan-400">{analysis.asymmetry}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Asymmetry</div>
              </div>

              <div className="text-center p-4 bg-green-50 rounded-lg dark:bg-green-950/30">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">{analysis.texture}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Texture Variance</div>
              </div>

              <div className="text-center p-4 bg-indigo-50 rounded-lg dark:bg-indigo-950/30">
                <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">{analysis.confidence}%</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Confidence</div>
              </div>
            </div>

            <div className="mt-4 p-4 bg-muted rounded-lg">
              <h3 className="font-semibold mb-2">Notes</h3>
              <p className="text-sm text-muted-foreground">
                These numbers are simulated for UI demonstration. They are not diagnostic. For medical decisions, see a
                qualified clinician.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
