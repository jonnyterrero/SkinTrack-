"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useSkinTrack } from "@/components/skintrack-provider"
import { EXPORT_SCHEMA_VERSION } from "@/lib/types"

export default function Integrations() {
  const { records, profile, repository, importBundle } = useSkinTrack()
  const [apiKey, setApiKey] = useState("")
  const [importData, setImportData] = useState("")
  const [exportedData, setExportedData] = useState("")
  const [webhookUrl, setWebhookUrl] = useState("")
  const [showSuccess, setShowSuccess] = useState(false)

  useEffect(() => {
    setApiKey(repository.getApiKey())
    setWebhookUrl(repository.getWebhookUrl())
  }, [repository])

  const generateApiKey = () => {
    const key = "sk_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
    setApiKey(key)
    repository.setApiKey(key)
    showSuccessMessage()
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
    showSuccessMessage()
  }

  const handleImport = async () => {
    try {
      const data = JSON.parse(importData) as unknown
      const result = await importBundle(data)
      if (!result.ok) {
        alert(result.error)
        return
      }
      setImportData("")
      showSuccessMessage()
    } catch {
      alert("Error parsing JSON. Please check the format and try again.")
    }
  }

  const copyApiEndpoint = () => {
    const endpoint = `${window.location.origin}/api/skintrack`
    void navigator.clipboard.writeText(endpoint)
    showSuccessMessage()
  }

  const saveWebhook = () => {
    repository.setWebhookUrl(webhookUrl)
    showSuccessMessage()
  }

  const showSuccessMessage = () => {
    setShowSuccess(true)
    setTimeout(() => setShowSuccess(false), 2000)
  }

  return (
    <div className="space-y-6">
      {showSuccess && (
        <div className="fixed top-4 right-4 z-50 glass-card rounded-xl p-4 border-green-200/30 bg-gradient-to-r from-green-500/20 to-emerald-500/20 shadow-2xl animate-in slide-in-from-top">
          <div className="text-green-700 text-sm font-medium flex items-center gap-2">
            <span className="text-green-500">✅</span>
            Success!
          </div>
        </div>
      )}

      <div>
        <h2 className="mb-2 bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-3xl font-bold text-transparent">
          Data &amp; integrations
        </h2>
        <p className="text-foreground/70">Export, import, and optional API hooks. Primary storage stays on your device.</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
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
            <div className="text-xs text-foreground/60 space-y-1">
              <p>GET returns integration metadata. POST echoes payloads only—no cloud persistence until Supabase is enabled.</p>
              <code className="block bg-black/20 p-2 rounded">Authorization: Bearer YOUR_API_KEY</code>
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
                  className="glass-input h-32 text-xs font-mono"
                />
                <p className="text-xs text-foreground/60">
                  {records.length} records exported. File downloaded automatically.
                </p>
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
                className="glass-input h-32 text-xs font-mono"
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
              <h4 className="font-semibold mb-2">GET /api/skintrack</h4>
              <p className="text-foreground/70 mb-2">Returns documentation JSON (no database reads).</p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">POST /api/skintrack</h4>
              <p className="text-foreground/70 mb-2">Validates body shape and echoes a synthetic record; does not write to Supabase yet.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
