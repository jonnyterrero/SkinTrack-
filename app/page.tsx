"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import ImageCapture from "@/components/image-capture"
import ImageAnalysis from "@/components/image-analysis"
import SymptomTracker from "@/components/symptom-tracker"
import DataAnalysis from "@/components/data-analysis"
import ImageGallery from "@/components/image-gallery"
import ProfileManager from "@/components/profile-manager"
import BodyMap from "@/components/body-map"

export default function HomePage() {
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null)
  const [isInstalled, setIsInstalled] = useState(false)
  const [records, setRecords] = useState<any[]>([])

  useEffect(() => {
    // Register service worker
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker
        .register("/sw.js")
        .then((registration) => {
          console.log("[v0] Service Worker registered:", registration)
        })
        .catch((error) => {
          console.log("[v0] Service Worker registration failed:", error)
        })
    }

    // Listen for install prompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault()
      setDeferredPrompt(e)
    }

    // Check if app is already installed
    const handleAppInstalled = () => {
      setIsInstalled(true)
      setDeferredPrompt(null)
    }

    window.addEventListener("beforeinstallprompt", handleBeforeInstallPrompt)
    window.addEventListener("appinstalled", handleAppInstalled)

    // Check if running in standalone mode (already installed)
    if (window.matchMedia("(display-mode: standalone)").matches) {
      setIsInstalled(true)
    }

    const savedRecords = localStorage.getItem("skintrack-records")
    if (savedRecords) {
      setRecords(JSON.parse(savedRecords))
    }

    return () => {
      window.removeEventListener("beforeinstallprompt", handleBeforeInstallPrompt)
      window.removeEventListener("appinstalled", handleAppInstalled)
    }
  }, [])

  const handleInstallClick = async () => {
    if (deferredPrompt) {
      deferredPrompt.prompt()
      const { outcome } = await deferredPrompt.userChoice
      console.log("[v0] Install prompt outcome:", outcome)
      setDeferredPrompt(null)
    }
  }

  const handleRecordSaved = (record: any) => {
    const newRecord = {
      ...record,
      id: Date.now(),
      timestamp: new Date().toISOString(),
    }
    const updatedRecords = [newRecord, ...records]
    setRecords(updatedRecords)
    localStorage.setItem("skintrack-records", JSON.stringify(updatedRecords))
  }

  const handleImageCaptured = (imageRecord: any) => {
    handleRecordSaved(imageRecord)
  }

  return (
    <div className="min-h-screen">
      <div className="container mx-auto p-4 relative">
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4 glass-card rounded-2xl p-6">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                SkinTrack+
              </h1>
              <p className="text-foreground/80 mt-2">Track and analyze your skin condition progress</p>
            </div>

            <div className="flex items-center gap-4">
              {!isInstalled && (
                <Button
                  onClick={
                    deferredPrompt
                      ? handleInstallClick
                      : () => {
                          // Fallback for browsers that don't support install prompt
                          alert(
                            "To install: \n\n1. Click the menu (⋮) in your browser\n2. Select 'Install SkinTrack+' or 'Add to Home Screen'\n3. Enjoy the native app experience!",
                          )
                        }
                  }
                  size="lg"
                  className="glass-button text-white border-4 border-yellow-400 hover:border-yellow-300 bg-gradient-to-r from-yellow-500 via-orange-500 to-red-500 hover:from-yellow-400 hover:via-orange-400 hover:to-red-400 px-8 py-4 text-lg font-black shadow-2xl hover:shadow-3xl transition-all transform hover:scale-110 animate-bounce"
                  style={{
                    boxShadow: "0 0 30px rgba(255, 193, 7, 0.8), 0 0 60px rgba(255, 152, 0, 0.6)",
                    textShadow: "2px 2px 4px rgba(0,0,0,0.7)",
                    animation: "bounce 1s infinite, pulse 2s infinite",
                  }}
                >
                  🚀 INSTALL FREE! 🚀
                </Button>
              )}
            </div>
          </div>

          {isInstalled && (
            <div className="glass-card rounded-xl p-4 mb-4 border-green-200/30 bg-gradient-to-r from-green-500/10 to-emerald-500/10">
              <div className="text-green-700 text-sm font-medium flex items-center gap-2">
                <span className="text-green-500">✅</span>
                App installed successfully! Enjoy the native experience.
              </div>
            </div>
          )}
        </div>

        <Tabs defaultValue="home" className="w-full">
          <TabsList className="glass-card grid w-full grid-cols-7 p-2 mb-6">
            <TabsTrigger
              value="home"
              className="glass-button data-[state=active]:bg-primary/80 data-[state=active]:text-white text-xs"
            >
              🏠 Home
            </TabsTrigger>
            <TabsTrigger
              value="profile"
              className="glass-button data-[state=active]:bg-primary/80 data-[state=active]:text-white text-xs"
            >
              👤 Profile
            </TabsTrigger>
            <TabsTrigger
              value="bodymap"
              className="glass-button data-[state=active]:bg-primary/80 data-[state=active]:text-white text-xs"
            >
              🗺️ Body Map
            </TabsTrigger>
            <TabsTrigger
              value="capture"
              className="glass-button data-[state=active]:bg-primary/80 data-[state=active]:text-white text-xs"
            >
              📸 Capture
            </TabsTrigger>
            <TabsTrigger
              value="analyze"
              className="glass-button data-[state=active]:bg-primary/80 data-[state=active]:text-white text-xs"
            >
              🔍 Analyze
            </TabsTrigger>
            <TabsTrigger
              value="track"
              className="glass-button data-[state=active]:bg-primary/80 data-[state=active]:text-white text-xs"
            >
              📝 Track
            </TabsTrigger>
            <TabsTrigger
              value="data"
              className="glass-button data-[state=active]:bg-primary/80 data-[state=active]:text-white text-xs"
            >
              📊 Data
            </TabsTrigger>
          </TabsList>

          <TabsContent value="home" className="space-y-6">
            <Card className="glass-card border-white/20">
              <CardHeader>
                <CardTitle className="text-2xl bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  Welcome to SkinTrack+
                </CardTitle>
                <CardDescription className="text-foreground/70">
                  Your comprehensive skin condition tracking companion
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h3 className="font-semibold text-lg">🎯 What can you do?</h3>
                    <ul className="space-y-3 text-sm">
                      <li className="flex items-center gap-3 p-2 rounded-lg glass hover:bg-white/10 transition-all">
                        <span className="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></span>
                        Capture and analyze skin condition images
                      </li>
                      <li className="flex items-center gap-3 p-2 rounded-lg glass hover:bg-white/10 transition-all">
                        <span className="w-3 h-3 bg-gradient-to-r from-green-500 to-teal-500 rounded-full"></span>
                        Track symptoms like itch, pain, and stress
                      </li>
                      <li className="flex items-center gap-3 p-2 rounded-lg glass hover:bg-white/10 transition-all">
                        <span className="w-3 h-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"></span>
                        Monitor progress over time
                      </li>
                      <li className="flex items-center gap-3 p-2 rounded-lg glass hover:bg-white/10 transition-all">
                        <span className="w-3 h-3 bg-gradient-to-r from-orange-500 to-red-500 rounded-full"></span>
                        Export data for healthcare providers
                      </li>
                    </ul>
                  </div>

                  <div className="space-y-4">
                    <h3 className="font-semibold text-lg">📊 Quick Stats</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-4 glass-card rounded-xl hover:scale-105 transition-transform">
                        <div className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                          {records.length}
                        </div>
                        <div className="text-xs text-foreground/60 mt-1">Total Records</div>
                      </div>
                      <div className="text-center p-4 glass-card rounded-xl hover:scale-105 transition-transform">
                        <div className="text-3xl font-bold bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
                          {records.filter((r) => r.type === "image").length}
                        </div>
                        <div className="text-xs text-foreground/60 mt-1">Images</div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="glass-card rounded-2xl p-1">
              <ImageGallery records={records} />
            </div>
          </TabsContent>

          <TabsContent value="profile">
            <div className="glass-card rounded-2xl p-6">
              <ProfileManager />
            </div>
          </TabsContent>

          <TabsContent value="bodymap">
            <div className="glass-card rounded-2xl p-6">
              <BodyMap records={records} />
            </div>
          </TabsContent>

          <TabsContent value="capture">
            <div className="glass-card rounded-2xl p-6">
              <ImageCapture onImageCaptured={handleImageCaptured} />
            </div>
          </TabsContent>

          <TabsContent value="analyze">
            <div className="glass-card rounded-2xl p-6">
              <ImageAnalysis />
            </div>
          </TabsContent>

          <TabsContent value="track">
            <div className="glass-card rounded-2xl p-6">
              <SymptomTracker onRecordSaved={handleRecordSaved} />
            </div>
          </TabsContent>

          <TabsContent value="data">
            <div className="glass-card rounded-2xl p-6">
              <DataAnalysis records={records} />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
