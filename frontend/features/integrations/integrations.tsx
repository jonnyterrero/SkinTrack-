"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useSkinTrack } from "@/components/skintrack-provider"
import { EXPORT_SCHEMA_VERSION } from "@/lib/types"
import { isSupabaseConfigured } from "@/lib/supabase/browser-client"
import { useAuth } from "@/context/AuthContext"
import { toast } from "sonner"
import { getSupabaseBrowserClient } from "@/lib/supabase/browser-client"

export default function Integrations() {
  const { records, profile, repository, importBundle, refresh, syncState, pendingCount, sync } = useSkinTrack()
  const { user, signOut } = useAuth()
  const [apiKey, setApiKey] = useState("")
  const [importData, setImportData] = useState("")
  const [exportedData, setExportedData] = useState("")
  const [webhookUrl, setWebhookUrl] = useState("")
  const [syncMessage, setSyncMessage] = useState<string | null>(null)
  const [authEmail, setAuthEmail] = useState("")

  useEffect(() => {
    setApiKey(repository.getApiKey())
    setWebhookUrl(repository.getWebhookUrl())
  }, [repository])

  const generateApiKey = () => {
    const key = "sk_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
    setApiKey(key)
    repository.setApiKey(key)
    toast.success("API key generated")
  }

  const handleExport = () => {
    const exportPayload = repository.buildExport(records, profile)
    const jsonString = JSON.stringify(exportPayload, null, 2)
    setExportedData(jsonString)

    const blob = new Blob([jsonString], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `skintrack-export-${new Date().toISOString().split("T")[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast.success("Export downloaded")
  }

  const handleImport = async () => {
    try {
      const data = JSON.parse(importData) as unknown
      const result = await importBundle(data)
      if (!result.ok) {
        toast.error(result.error)
        return
      }
      setImportData("")
      await refresh()
      toast.success("Import complete")
    } catch {
      toast.error("Invalid JSON. Check the format and try again.")
    }
  }

  const handleSyncNow = async () => {
    setSyncMessage(null)
    const result = await sync()
    if (result === "synced") {
      toast.success("Synced to cloud")
    } else if (result === "error") {
      setSyncMessage("Sync failed — will retry automatically.")
      toast.error("Sync failed")
    } else if (result === "dirty") {
      toast.info("Partial sync — more items pending")
    }
  }

  const sendMagicLink = async () => {
    const supabase = getSupabaseBrowserClient()
    if (!supabase || !authEmail.trim()) return
    const { error } = await supabase.auth.signInWithOtp({
      email: authEmail.trim(),
      options: { emailRedirectTo: typeof window !== "undefined" ? `${window.location.origin}/auth/callback` : undefined },
    })
    if (error) {
      setSyncMessage(error.message)
      return
    }
    setSyncMessage("Check your email for the login link.")
  }

  const copyApiEndpoint = () => {
    const endpoint = `${window.location.origin}/api/skintrack`
    void navigator.clipboard.writeText(endpoint)
    toast.success("Endpoint copied")
  }

  const saveWebhook = () => {
    repository.setWebhookUrl(webhookUrl)
    toast.success("Webhook URL saved (not sent automatically)")
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="mb-2 bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-3xl font-bold text-transparent">
          Data &amp; integrations
        </h2>
        <p className="text-foreground/70">Export, import, optional cloud sync, and API hooks. Primary storage stays on your device.</p>
      </div>

      {isSupabaseConfigured() ? (
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="text-lg">Cloud sync (Supabase)</CardTitle>
            <CardDescription>
              Sign in with a one-time email link to enable automatic cloud backup. Your data syncs every 30 seconds when signed in.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {user ? (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  Signed in as <strong>{user.email}</strong>
                </p>
                <p className="text-sm text-muted-foreground">
                  Sync status: <strong>{syncState}</strong>
                  {pendingCount > 0 && ` — ${pendingCount} pending`}
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-2 sm:flex-row">
                <Input
                  type="email"
                  placeholder="you@example.com"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  className="glass-input"
                />
                <Button type="button" variant="secondary" onClick={() => void sendMagicLink()}>
                  Send magic link
                </Button>
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              <Button type="button" onClick={() => void handleSyncNow()} disabled={syncState === "syncing" || !user}>
                {syncState === "syncing" ? "Syncing..." : "Sync now"}
              </Button>
              {user ? (
                <Button type="button" variant="outline" onClick={() => void signOut()}>
                  Sign out
                </Button>
              ) : null}
            </div>
            {syncMessage ? <p className="text-sm text-muted-foreground">{syncMessage}</p> : null}
          </CardContent>
        </Card>
      ) : (
        <Card className="border-dashed border-slate-300 dark:border-slate-600">
          <CardHeader>
            <CardTitle className="text-lg">Supabase</CardTitle>
            <CardDescription>
              Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in your environment to enable cloud sync.
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>🔑</span> API Key
            </CardTitle>
            <CardDescription>For future server routes; data today is local-first</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Your API Key</Label>
              <div className="flex gap-2">
                <Input
                  type="password"
                  value={apiKey}
                  readOnly
                  placeholder="No API key generated"
                  className="glass-input"
                />
                <Button onClick={generateApiKey} className="glass-button">
                  Generate
                </Button>
              </div>
            </div>
            <div className="space-y-2">
              <Label>API Endpoint</Label>
              <div className="flex gap-2">
                <Input
                  value={`${typeof window !== "undefined" ? window.location.origin : ""}/api/skintrack`}
                  readOnly
                  className="glass-input text-xs"
                />
                <Button onClick={copyApiEndpoint} className="glass-button">
                  Copy
                </Button>
              </div>
            </div>
            <div className="space-y-1 text-xs text-foreground/60">
              <p>GET returns integration metadata. POST echoes payloads only—no cloud persistence until Supabase is enabled.</p>
              <code className="block rounded bg-black/20 p-2">Authorization: Bearer YOUR_API_KEY</code>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>📤</span> Export Data
            </CardTitle>
            <CardDescription>Versioned JSON (v{EXPORT_SCHEMA_VERSION}) including profile</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={handleExport} className="w-full glass-button">
              Export All Data
            </Button>
            {exportedData && (
              <div className="space-y-2">
                <Label>Exported Data Preview</Label>
                <Textarea
                  value={exportedData.substring(0, 200) + "..."}
                  readOnly
                  className="glass-input h-32 font-mono text-xs"
                />
                <p className="text-xs text-foreground/60">{records.length} records exported. File downloaded automatically.</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>📥</span> Import Data
            </CardTitle>
            <CardDescription>Merges with existing local records</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Paste JSON Data</Label>
              <Textarea
                value={importData}
                onChange={(e) => setImportData(e.target.value)}
                placeholder={`{"version": ${EXPORT_SCHEMA_VERSION}, "records": [...], "profile": {...}}`}
                className="glass-input h-32 font-mono text-xs"
              />
            </div>
            <Button onClick={() => void handleImport()} className="w-full glass-button" disabled={!importData}>
              Import Data
            </Button>
            <p className="text-xs text-foreground/60">Supports v1 exports and legacy version &quot;1.0&quot; bundles.</p>
          </CardContent>
        </Card>

        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>🔔</span> Webhooks
            </CardTitle>
            <CardDescription>Stored locally for future automation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Webhook URL</Label>
              <Input
                type="url"
                value={webhookUrl}
                onChange={(e) => setWebhookUrl(e.target.value)}
                placeholder="https://your-app.com/webhook"
                className="glass-input"
              />
            </div>
            <Button onClick={saveWebhook} className="w-full glass-button" disabled={!webhookUrl}>
              Save Webhook
            </Button>
            <p className="text-xs text-foreground/60">Delivery is not active in this build; URL is saved for upcoming sync features.</p>
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span>📚</span> API notes
          </CardTitle>
          <CardDescription>How the Next.js route behaves today</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3 text-sm">
            <div>
              <h4 className="mb-2 font-semibold">GET /api/skintrack</h4>
              <p className="mb-2 text-foreground/70">Returns documentation JSON (no database reads).</p>
            </div>
            <div>
              <h4 className="mb-2 font-semibold">POST /api/skintrack</h4>
              <p className="mb-2 text-foreground/70">Validates body shape and echoes a synthetic record; use Supabase sync for cloud persistence.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
