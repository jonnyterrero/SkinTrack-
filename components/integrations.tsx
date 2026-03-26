"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"

interface IntegrationsProps {
  records: any[]
  onRecordsImported: (records: any[]) => void
}

export default function Integrations({ records, onRecordsImported }: IntegrationsProps) {
  const [apiKey, setApiKey] = useState("")
  const [importData, setImportData] = useState("")
  const [exportedData, setExportedData] = useState("")
  const [webhookUrl, setWebhookUrl] = useState("")
  const [showSuccess, setShowSuccess] = useState(false)

  // Generate a simple API key
  const generateApiKey = () => {
    const key = "sk_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
    setApiKey(key)
    localStorage.setItem("skintrack-api-key", key)
    showSuccessMessage()
  }

  // Export all data as JSON
  const handleExport = () => {
    const exportData = {
      version: "1.0",
      exportDate: new Date().toISOString(),
      records: records,
      profile: JSON.parse(localStorage.getItem("skintrack-profile") || "{}"),
    }
    const jsonString = JSON.stringify(exportData, null, 2)
    setExportedData(jsonString)

    // Download as file
    const blob = new Blob([jsonString], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `skintrack-export-${new Date().toISOString().split("T")[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    showSuccessMessage()
  }

  // Import data from JSON
  const handleImport = () => {
    try {
      const data = JSON.parse(importData)
      if (data.records && Array.isArray(data.records)) {
        const mergedRecords = [...records, ...data.records]
        onRecordsImported(mergedRecords)
        localStorage.setItem("skintrack-records", JSON.stringify(mergedRecords))

        if (data.profile) {
          localStorage.setItem("skintrack-profile", JSON.stringify(data.profile))
        }

        setImportData("")
        showSuccessMessage()
      } else {
        alert('Invalid data format. Please ensure the JSON contains a "records" array.')
      }
    } catch (error) {
      alert("Error parsing JSON. Please check the format and try again.")
    }
  }

  // Copy API endpoint URL
  const copyApiEndpoint = () => {
    const endpoint = `${window.location.origin}/api/skintrack`
    navigator.clipboard.writeText(endpoint)
    showSuccessMessage()
  }

  // Save webhook URL
  const saveWebhook = () => {
    localStorage.setItem("skintrack-webhook-url", webhookUrl)
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
          App Integrations
        </h2>
        <p className="text-foreground/70">Connect SkinTrack+ with your other applications</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* API Key Management */}
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>🔑</span> API Key
            </CardTitle>
            <CardDescription>Generate an API key for programmatic access</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Your API Key</Label>
              <div className="flex gap-2">
                <Input
                  type="password"
                  value={apiKey || localStorage.getItem("skintrack-api-key") || ""}
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
              <p>Use this API key in your requests:</p>
              <code className="block bg-black/20 p-2 rounded">Authorization: Bearer YOUR_API_KEY</code>
            </div>
          </CardContent>
        </Card>

        {/* Export Data */}
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>📤</span> Export Data
            </CardTitle>
            <CardDescription>Download your data as JSON</CardDescription>
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

        {/* Import Data */}
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>📥</span> Import Data
            </CardTitle>
            <CardDescription>Import data from another app or backup</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Paste JSON Data</Label>
              <Textarea
                value={importData}
                onChange={(e) => setImportData(e.target.value)}
                placeholder='{"records": [...], "profile": {...}}'
                className="glass-input h-32 text-xs font-mono"
              />
            </div>
            <Button onClick={handleImport} className="w-full glass-button" disabled={!importData}>
              Import Data
            </Button>
            <p className="text-xs text-foreground/60">Paste exported JSON data from SkinTrack+ or compatible apps</p>
          </CardContent>
        </Card>

        {/* Webhook Configuration */}
        <Card className="glass-card border-slate-200/80 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>🔔</span> Webhooks
            </CardTitle>
            <CardDescription>Send data to external services</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Webhook URL</Label>
              <Input
                type="url"
                value={webhookUrl || localStorage.getItem("skintrack-webhook-url") || ""}
                onChange={(e) => setWebhookUrl(e.target.value)}
                placeholder="https://your-app.com/webhook"
                className="glass-input"
              />
            </div>
            <Button onClick={saveWebhook} className="w-full glass-button" disabled={!webhookUrl}>
              Save Webhook
            </Button>
            <p className="text-xs text-foreground/60">New records will be sent to this URL automatically</p>
          </CardContent>
        </Card>
      </div>

      {/* API Documentation */}
      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span>📚</span> API Documentation
          </CardTitle>
          <CardDescription>How to integrate with SkinTrack+</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3 text-sm">
            <div>
              <h4 className="font-semibold mb-2">GET /api/skintrack</h4>
              <p className="text-foreground/70 mb-2">Retrieve all records</p>
              <code className="block bg-black/20 p-3 rounded text-xs">
                {`fetch('${typeof window !== "undefined" ? window.location.origin : ""}/api/skintrack', {
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY'
  }
})`}
              </code>
            </div>

            <div>
              <h4 className="font-semibold mb-2">POST /api/skintrack</h4>
              <p className="text-foreground/70 mb-2">Add a new record</p>
              <code className="block bg-black/20 p-3 rounded text-xs">
                {`fetch('${typeof window !== "undefined" ? window.location.origin : ""}/api/skintrack', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    type: 'symptom',
    data: { ... }
  })
})`}
              </code>
            </div>

            <div className="pt-2 border-t border-white/10">
              <h4 className="font-semibold mb-2">Data Format</h4>
              <p className="text-foreground/70 mb-2">Records follow this structure:</p>
              <code className="block bg-black/20 p-3 rounded text-xs">
                {`{
  "id": 1234567890,
  "timestamp": "2024-01-01T00:00:00.000Z",
  "type": "image" | "symptom",
  "data": { ... }
}`}
              </code>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
