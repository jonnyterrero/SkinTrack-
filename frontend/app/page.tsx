"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import {
  BookOpen,
  ChevronLeft,
  Cloud,
  CloudOff,
  Home,
  LineChart,
  Link2,
  Loader2,
  LogIn,
  LogOut,
  ScanLine,
  SlidersHorizontal,
  User,
} from "lucide-react"
import { useAuth } from "@/context/AuthContext"
import { useReminders } from "@/hooks/useReminders"
import { apiGet } from "@/lib/api/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import ImageCapture from "@/features/images/image-capture"
import ImageAnalysis from "@/features/images/image-analysis"
import SymptomTracker from "@/features/symptom/symptom-tracker"
import SkinEventLog from "@/features/skin-event/skin-event-log"
import DataAnalysis from "@/features/data/data-analysis"
import ImageGallery from "@/components/image-gallery"
import ProfileManager from "@/features/profile/profile-manager"
import BodyMap from "@/features/body-map/body-map"
import Integrations from "@/features/integrations/integrations"
import ProductSetup from "@/features/setup/product-setup"
import AboutSkinTrack from "@/components/about-skintrack"
import { StorageErrorBanner } from "@/components/storage-error-banner"
import { useSkinTrack } from "@/components/skintrack-provider"
import { cn } from "@/lib/utils"
import type { ImageMetadata, NewSkinTrackRecordInput } from "@/lib/types"
import { toast } from "sonner"

const navTriggerClass =
  "flex h-auto flex-col items-center gap-1 rounded-lg p-2 text-gray-600 transition-all duration-200 hover:bg-gray-50 hover:text-cyan-600 data-[state=active]:bg-cyan-50 data-[state=active]:text-cyan-600 dark:text-gray-400 dark:hover:bg-slate-800 dark:hover:text-cyan-400 dark:data-[state=active]:bg-cyan-950/40 dark:data-[state=active]:text-cyan-400"

export default function HomePage() {
  const { records, loading, storageError, clearStorageError, saveRecord, syncState, pendingCount, sync } = useSkinTrack()
  const { user, signOut } = useAuth()
  const router = useRouter()
  useReminders()
  const [showLegacySymptomLog, setShowLegacySymptomLog] = useState(false)
  const [deferredPrompt, setDeferredPrompt] = useState<unknown>(null)
  const [isInstalled, setIsInstalled] = useState(false)
  const [showUpdateNotification, setShowUpdateNotification] = useState(false)
  const [tab, setTab] = useState("home")

  useEffect(() => {
    if (!user) return
    apiGet<{ completed_onboarding: boolean }>("/api/app-preferences")
      .then((p) => {
        if (!p.completed_onboarding) router.replace("/onboarding")
      })
      .catch(() => {})
  }, [user, router])

  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker
        .register("/sw.js")
        .then((registration) => {
          console.log("[SkinTrack+] Service Worker registered:", registration)

          setInterval(() => {
            registration.update()
          }, 60000)

          registration.addEventListener("updatefound", () => {
            const newWorker = registration.installing
            if (newWorker) {
              newWorker.addEventListener("statechange", () => {
                if (newWorker.state === "activated" && navigator.serviceWorker.controller) {
                  setShowUpdateNotification(true)
                  setTimeout(() => {
                    window.location.reload()
                  }, 1500)
                }
              })
            }
          })
        })
        .catch((error) => {
          console.log("[SkinTrack+] Service Worker registration failed:", error)
        })

      navigator.serviceWorker.addEventListener("controllerchange", () => {
        console.log("[SkinTrack+] New service worker activated, reloading...")
        setShowUpdateNotification(true)
        setTimeout(() => {
          window.location.reload()
        }, 1500)
      })
    }

    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault()
      setDeferredPrompt(e)
    }

    const handleAppInstalled = () => {
      setIsInstalled(true)
      setDeferredPrompt(null)
    }

    window.addEventListener("beforeinstallprompt", handleBeforeInstallPrompt)
    window.addEventListener("appinstalled", handleAppInstalled)

    if (window.matchMedia("(display-mode: standalone)").matches) {
      setIsInstalled(true)
    }

    return () => {
      window.removeEventListener("beforeinstallprompt", handleBeforeInstallPrompt)
      window.removeEventListener("appinstalled", handleAppInstalled)
    }
  }, [])

  const handleInstallClick = async () => {
    const prompt = deferredPrompt as { prompt?: () => void; userChoice?: Promise<{ outcome: string }> } | null
    if (prompt?.prompt) {
      prompt.prompt()
      const { outcome } = await (prompt.userChoice ?? Promise.resolve({ outcome: "dismissed" }))
      console.log("[SkinTrack+] Install prompt outcome:", outcome)
      setDeferredPrompt(null)
    }
  }

  const handleRecordSaved = async (input: NewSkinTrackRecordInput) => {
    await saveRecord(input)
  }

  const handleImageCaptured = (payload: { type: "image"; filename: string; image: string; metadata?: ImageMetadata }) => {
    void handleRecordSaved(payload)
  }

  return (
    <div className="st-app-shell">
      <Tabs value={tab} onValueChange={setTab} className="flex min-h-screen flex-col gap-0">
        <div className="relative container mx-auto flex-1 px-4 py-6 pb-28">
          {showUpdateNotification && (
            <div
              className="fixed top-4 left-1/2 z-[60] -translate-x-1/2 rounded-xl border border-white/30 px-4 py-3 text-center text-sm font-medium text-white shadow-lg animate-in slide-in-from-top"
              style={{
                background: "linear-gradient(135deg, rgba(8, 145, 178, 0.95), rgba(6, 182, 212, 0.95))",
                backdropFilter: "blur(20px)",
              }}
            >
              <div className="flex items-center justify-center gap-2">
                <span aria-hidden>🔄</span>
                New version available! Updating app...
              </div>
            </div>
          )}

          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              {tab !== "home" && (
                <button
                  type="button"
                  onClick={() => setTab("home")}
                  className="rounded-full border border-white/20 bg-white/80 p-2 shadow-lg backdrop-blur-sm transition-all hover:bg-white/90 dark:border-slate-600 dark:bg-slate-800/80 dark:hover:bg-slate-800"
                  aria-label="Back to home"
                >
                  <ChevronLeft className="h-5 w-5 text-gray-600 dark:text-gray-300" />
                </button>
              )}
              <div>
                <h1 className="bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-2xl font-bold text-transparent">
                  SkinTrack+
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">Skin condition tracker</p>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              {user && (
                <button
                  type="button"
                  onClick={() => void sync()}
                  className="relative rounded-full p-2 text-gray-500 transition-colors hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-slate-800"
                  title={`Sync: ${syncState}${pendingCount > 0 ? ` (${pendingCount} pending)` : ""}`}
                >
                  {syncState === "syncing" ? (
                    <Loader2 className="h-5 w-5 animate-spin text-cyan-500" />
                  ) : syncState === "error" ? (
                    <CloudOff className="h-5 w-5 text-red-500" />
                  ) : (
                    <Cloud className="h-5 w-5 text-emerald-500" />
                  )}
                  {pendingCount > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-cyan-500 text-[10px] font-bold text-white">
                      {pendingCount > 9 ? "9+" : pendingCount}
                    </span>
                  )}
                </button>
              )}
              {user ? (
                <Button
                  onClick={() => void signOut()}
                  size="sm"
                  variant="outline"
                  className="gap-1.5 rounded-lg px-3 py-2 text-sm"
                >
                  <LogOut className="h-4 w-4" />
                  <span className="hidden sm:inline">Sign out</span>
                </Button>
              ) : (
                <Button
                  onClick={() => setTab("integrations")}
                  size="sm"
                  variant="outline"
                  className="gap-1.5 rounded-lg px-3 py-2 text-sm"
                >
                  <LogIn className="h-4 w-4" />
                  <span className="hidden sm:inline">Sign in</span>
                </Button>
              )}
              {!isInstalled && (
                <Button
                  onClick={
                    deferredPrompt
                      ? handleInstallClick
                      : () => {
                          toast.info("Install SkinTrack+", {
                            description:
                              "Open the browser menu (⋮), then choose Install or Add to Home Screen. Open the app from your home screen.",
                            duration: 10_000,
                          })
                        }
                  }
                  size="sm"
                  className="glass-button rounded-lg border-0 px-4 py-2 text-sm font-semibold text-white shadow-md"
                >
                  Install
                </Button>
              )}
            </div>
          </div>

          {user && (
            <div className="mb-4 -mt-2 flex flex-wrap gap-1.5 text-xs">
              <Link href="/medications" className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">Meds &amp; remedies</Link>
              <Link href="/body-map" className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">Body map</Link>
              <Link href="/export" className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">Export</Link>
              <Link href="/settings" className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">Settings</Link>
              <Link href="/about" className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">About</Link>
            </div>
          )}

          <StorageErrorBanner message={storageError} onDismiss={clearStorageError} />

          {loading ? (
            <div className="glass-card rounded-2xl border border-slate-200/80 p-8 text-center dark:border-slate-700">
              <p className="text-muted-foreground">Loading your local data…</p>
            </div>
          ) : null}

          {!loading && isInstalled && (
            <div className="mb-4 rounded-xl border border-emerald-200/80 bg-emerald-50/90 p-3 dark:border-emerald-800 dark:bg-emerald-950/40">
              <div className="flex items-center gap-2 text-sm font-medium text-emerald-800 dark:text-emerald-300">
                <span className="text-emerald-600 dark:text-emerald-400" aria-hidden>
                  ✓
                </span>
                App installed — use it like a native app.
              </div>
            </div>
          )}

          {!loading && records.length === 0 ? (
            <div className="mb-6 rounded-xl border border-dashed border-cyan-200/80 bg-white/60 p-4 text-center text-sm text-muted-foreground dark:border-cyan-900/50 dark:bg-slate-900/40">
              No entries yet. Add a symptom log or save a photo from the <strong>Scan</strong> tab, or import a backup
              from <strong>Data</strong> (integrations).
            </div>
          ) : null}

          {!loading ? (
            <>
          <TabsContent value="home" className="mt-0 space-y-6 outline-none focus-visible:outline-none">
            <Card className="glass-card border-slate-200/80 dark:border-slate-700">
              <CardHeader>
                <CardTitle className="bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-2xl text-transparent">
                  Welcome to SkinTrack+
                </CardTitle>
                <CardDescription className="text-muted-foreground">
                  Your comprehensive skin condition tracking companion
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold">What you can do</h3>
                    <ul className="space-y-3 text-sm">
                      <li className="glass flex items-center gap-3 rounded-lg p-2 transition-colors hover:bg-white/60 dark:hover:bg-slate-800/60">
                        <span className="h-3 w-3 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500" />
                        Capture and analyze skin condition images
                      </li>
                      <li className="glass flex items-center gap-3 rounded-lg p-2 transition-colors hover:bg-white/60 dark:hover:bg-slate-800/60">
                        <span className="h-3 w-3 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500" />
                        Track symptoms like itch, pain, and stress
                      </li>
                      <li className="glass flex items-center gap-3 rounded-lg p-2 transition-colors hover:bg-white/60 dark:hover:bg-slate-800/60">
                        <span className="h-3 w-3 rounded-full bg-gradient-to-r from-orange-400 to-orange-600" />
                        Monitor progress over time
                      </li>
                      <li className="glass flex items-center gap-3 rounded-lg p-2 transition-colors hover:bg-white/60 dark:hover:bg-slate-800/60">
                        <span className="h-3 w-3 rounded-full bg-gradient-to-r from-sky-500 to-cyan-600" />
                        Export data for healthcare providers
                      </li>
                    </ul>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold">Quick stats</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="glass-card rounded-xl p-4 text-center transition-transform hover:scale-[1.02]">
                        <div className="bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-3xl font-bold text-transparent">
                          {records.length}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">Total records</div>
                      </div>
                      <div className="glass-card rounded-xl p-4 text-center transition-transform hover:scale-[1.02]">
                        <div className="bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-3xl font-bold text-transparent">
                          {records.filter((r) => r.type === "image").length}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">Images</div>
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

          <TabsContent value="profile" className="mt-0 outline-none focus-visible:outline-none">
            <div className="glass-card rounded-2xl p-6">
              <ProfileManager />
            </div>
          </TabsContent>

          <TabsContent value="scan" className="mt-0 space-y-6 outline-none focus-visible:outline-none">
            <div className="glass-card rounded-2xl p-6">
              <BodyMap records={records} />
            </div>
            <div className="glass-card rounded-2xl p-6">
              <ImageCapture onImageCaptured={handleImageCaptured} />
            </div>
            <div className="glass-card rounded-2xl p-6">
              <ImageAnalysis />
            </div>
          </TabsContent>

          <TabsContent value="insights" className="mt-0 space-y-6 outline-none focus-visible:outline-none">
            <div className="glass-card rounded-2xl p-6">
              <SkinEventLog />
            </div>
            <div className="rounded-xl border border-dashed border-slate-300/80 p-4 dark:border-slate-600">
              <button
                type="button"
                className="w-full text-left text-sm font-medium text-muted-foreground hover:text-foreground"
                onClick={() => setShowLegacySymptomLog((v) => !v)}
              >
                {showLegacySymptomLog ? "▼" : "▶"} Previous log format (legacy)
              </button>
              {showLegacySymptomLog ? (
                <div className="mt-4">
                  <SymptomTracker onRecordSaved={handleRecordSaved} />
                </div>
              ) : null}
            </div>
            <div className="glass-card rounded-2xl p-6">
              <DataAnalysis records={records} />
            </div>
          </TabsContent>

          <TabsContent value="setup" className="mt-0 outline-none focus-visible:outline-none">
            <div className="glass-card rounded-2xl p-6">
              <ProductSetup />
            </div>
          </TabsContent>

          <TabsContent value="integrations" className="mt-0 outline-none focus-visible:outline-none">
            <div className="glass-card rounded-2xl p-6">
              <Integrations />
            </div>
          </TabsContent>

          <TabsContent value="about" className="mt-0 outline-none focus-visible:outline-none">
            <div className="glass-card rounded-2xl p-6">
              <AboutSkinTrack />
            </div>
          </TabsContent>
            </>
          ) : null}
        </div>

        <div
          className={cn(
            "fixed bottom-0 left-0 right-0 z-50 border-t border-gray-200 bg-white/90 backdrop-blur-sm dark:border-slate-700 dark:bg-slate-900/90",
            "pb-[max(0.5rem,env(safe-area-inset-bottom))]",
          )}
        >
          <TabsList
            className={cn(
              "mx-auto grid h-auto w-full max-w-3xl grid-cols-7 gap-0 rounded-none border-0 bg-transparent px-1 py-2 text-muted-foreground shadow-none backdrop-blur-none sm:px-4",
            )}
          >
            <TabsTrigger value="home" className={cn(navTriggerClass, "!shadow-none data-[state=active]:!bg-cyan-50 dark:data-[state=active]:!bg-cyan-950/40")}>
              <Home className="h-5 w-5 shrink-0" aria-hidden />
              <span className="text-xs font-medium">Home</span>
            </TabsTrigger>
            <TabsTrigger value="profile" className={cn(navTriggerClass, "!shadow-none data-[state=active]:!bg-cyan-50 dark:data-[state=active]:!bg-cyan-950/40")}>
              <User className="h-5 w-5 shrink-0" aria-hidden />
              <span className="text-xs font-medium">Profile</span>
            </TabsTrigger>
            <TabsTrigger value="scan" className={cn(navTriggerClass, "!shadow-none data-[state=active]:!bg-cyan-50 dark:data-[state=active]:!bg-cyan-950/40")}>
              <ScanLine className="h-5 w-5 shrink-0" aria-hidden />
              <span className="text-xs font-medium">Scan</span>
            </TabsTrigger>
            <TabsTrigger value="insights" className={cn(navTriggerClass, "!shadow-none data-[state=active]:!bg-cyan-50 dark:data-[state=active]:!bg-cyan-950/40")}>
              <LineChart className="h-5 w-5 shrink-0" aria-hidden />
              <span className="text-[10px] font-medium sm:text-xs">Insights</span>
            </TabsTrigger>
            <TabsTrigger value="setup" className={cn(navTriggerClass, "!shadow-none data-[state=active]:!bg-cyan-50 dark:data-[state=active]:!bg-cyan-950/40")}>
              <SlidersHorizontal className="h-5 w-5 shrink-0" aria-hidden />
              <span className="text-[10px] font-medium sm:text-xs">Setup</span>
            </TabsTrigger>
            <TabsTrigger
              value="integrations"
              className={cn(navTriggerClass, "!shadow-none data-[state=active]:!bg-cyan-50 dark:data-[state=active]:!bg-cyan-950/40")}
            >
              <Link2 className="h-5 w-5 shrink-0" aria-hidden />
              <span className="text-[10px] font-medium leading-tight sm:text-xs">Data</span>
            </TabsTrigger>
            <TabsTrigger value="about" className={cn(navTriggerClass, "!shadow-none data-[state=active]:!bg-cyan-50 dark:data-[state=active]:!bg-cyan-950/40")}>
              <BookOpen className="h-5 w-5 shrink-0" aria-hidden />
              <span className="text-[10px] font-medium sm:text-xs">About</span>
            </TabsTrigger>
          </TabsList>
        </div>
      </Tabs>
    </div>
  )
}
