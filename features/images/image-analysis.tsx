"use client"

import { useState, type ChangeEvent } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScanIcon as Analyze, Upload } from "lucide-react"
import { isUnetAnalysisEnabled } from "@/lib/domain/unet"
import { Badge } from "@/components/ui/badge"

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
  const unet = isUnetAnalysisEnabled()

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
      <Card className="border-amber-300/80 bg-amber-50/90 dark:border-amber-800/60 dark:bg-amber-950/20">
        <CardHeader className="pb-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="bg-amber-200/80 text-amber-950 dark:bg-amber-900/40 dark:text-amber-100">
              Demo only
            </Badge>
            <span className="text-sm font-medium text-amber-950 dark:text-amber-100">Not a medical device</span>
          </div>
          <CardDescription className="text-amber-950/90 dark:text-amber-100/80">
            Numbers below are generated for UI testing. They do not analyze your skin and must not be used for
            diagnosis or treatment. Use saved photos and your clinician for real assessment.
          </CardDescription>
        </CardHeader>
      </Card>

      {unet ? (
        <Card className="border-amber-200/60 bg-amber-50/50 dark:border-amber-900/40 dark:bg-amber-950/20">
          <CardHeader>
            <CardTitle className="text-base">Experimental segmentation</CardTitle>
            <CardDescription>
              NEXT_PUBLIC_ENABLE_UNET is on. A U-Net or similar model can be wired here once the baseline image pipeline
              is stable; metrics below remain demo-only until then.
            </CardDescription>
          </CardHeader>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload image (demo)
          </CardTitle>
          <CardDescription>
            For layout testing only. No image processing runs in the browser in this build beyond the simulated
            delay below.
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
              <div className="overflow-hidden rounded-lg border">
                <img
                  src={selectedImage || "/placeholder.svg"}
                  alt="Image for analysis"
                  className="h-auto max-h-96 w-full object-contain"
                />
              </div>

              <Button onClick={analyzeImage} disabled={analyzing} className="w-full">
                <Analyze className="mr-2 h-4 w-4" />
                {analyzing ? "Running demo…" : "Run demo analysis"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {analysis && (
        <Card>
          <CardHeader>
            <CardTitle>Simulated metrics (random)</CardTitle>
            <CardDescription>
              Placeholder values for layout and flow testing. Labels do not imply a real algorithm.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
              <div className="rounded-lg bg-blue-50 p-4 text-center dark:bg-blue-950/30">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{analysis.area}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Fake area (cm²)</div>
              </div>

              <div className="rounded-lg bg-red-50 p-4 text-center dark:bg-red-950/30">
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">{analysis.redness}%</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Fake redness</div>
              </div>

              <div className="rounded-lg bg-yellow-50 p-4 text-center dark:bg-yellow-950/30">
                <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{analysis.irregularity}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Fake irregularity</div>
              </div>

              <div className="rounded-lg bg-cyan-50 p-4 text-center dark:bg-cyan-950/40">
                <div className="text-2xl font-bold text-cyan-700 dark:text-cyan-400">{analysis.asymmetry}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Fake asymmetry</div>
              </div>

              <div className="rounded-lg bg-green-50 p-4 text-center dark:bg-green-950/30">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">{analysis.texture}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Fake texture</div>
              </div>

              <div className="rounded-lg bg-indigo-50 p-4 text-center dark:bg-indigo-950/30">
                <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">{analysis.confidence}%</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Fake confidence</div>
              </div>
            </div>

            <div className="mt-4 rounded-lg bg-muted p-4">
              <h3 className="mb-2 font-semibold">Disclaimer</h3>
              <p className="text-sm text-muted-foreground">
                These numbers are random and not diagnostic. For medical decisions, see a qualified clinician.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
