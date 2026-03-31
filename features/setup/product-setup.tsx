"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BODY_AREA_DEFINITIONS } from "@/lib/domain/body-areas"
import { SKIN_CONDITIONS } from "@/lib/domain/conditions"
import { SCALE_ZERO_TO_TEN, SLEEP_HOURS, SYMPTOM_SCALE_LABELS } from "@/lib/domain/scales"
import { SEVERITY_DESCRIPTIONS, SEVERITY_ORDER } from "@/lib/domain/severity"

export default function ProductSetup() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="mb-2 bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-3xl font-bold text-transparent">
          Clinical setup
        </h2>
        <p className="text-foreground/70">
          Canonical scales and lists used across logging, the body map, and exports. These definitions keep the app
          consistent as features grow.
        </p>
      </div>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Symptom scales</CardTitle>
          <CardDescription>How entries are interpreted everywhere in SkinTrack+</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <ul className="space-y-3 text-sm">
            {(["itch", "pain", "stress"] as const).map((key) => (
              <li key={key} className="rounded-lg border border-slate-200/80 p-3 dark:border-slate-700">
                <span className="font-semibold">{SYMPTOM_SCALE_LABELS[key]}</span>
                <span className="text-muted-foreground">
                  {" "}
                  — {SCALE_ZERO_TO_TEN.label}: {SCALE_ZERO_TO_TEN.description}
                </span>
              </li>
            ))}
            <li className="rounded-lg border border-slate-200/80 p-3 dark:border-slate-700">
              <span className="font-semibold">Sleep</span>
              <span className="text-muted-foreground">
                {" "}
                — {SLEEP_HOURS.label} ({SLEEP_HOURS.min}–{SLEEP_HOURS.max}): {SLEEP_HOURS.description}
              </span>
            </li>
          </ul>
        </CardContent>
      </Card>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Severity (body map)</CardTitle>
          <CardDescription>Color coding on the interactive map</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {SEVERITY_ORDER.map((s) => (
            <div key={s} className="text-sm">
              <span className="font-medium capitalize">{s}</span>
              <span className="text-muted-foreground"> — {SEVERITY_DESCRIPTIONS[s]}</span>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Condition types</CardTitle>
          <CardDescription>Picker options on symptom logs</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-wrap gap-2">
            {SKIN_CONDITIONS.map((c) => (
              <li
                key={c}
                className="rounded-full border border-cyan-200/60 bg-cyan-50/80 px-3 py-1 text-xs dark:border-cyan-900/50 dark:bg-cyan-950/30"
              >
                {c}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Body regions</CardTitle>
          <CardDescription>IDs shared by the symptom form and body map ({BODY_AREA_DEFINITIONS.length} areas)</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="grid gap-1 text-sm sm:grid-cols-2">
            {BODY_AREA_DEFINITIONS.map((a) => (
              <li key={a.id}>
                <code className="text-xs text-muted-foreground">{a.id}</code> — {a.name}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
