"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function AboutSkinTrack() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="mb-2 bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-3xl font-bold text-transparent">
          About SkinTrack+
        </h2>
        <p className="text-foreground/70">
          Purpose, how to read the scales, and what this app is (and is not) for clinical conversations.
        </p>
      </div>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Purpose</CardTitle>
          <CardDescription>Local-first tracking for chronic skin conditions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-foreground/80">
          <p>
            SkinTrack+ helps you turn <strong>daily symptoms</strong> and <strong>photos over time</strong> into a
            structured history you can review yourself or share with a clinician. It supports eczema, psoriasis,
            acne, vitiligo, contact dermatitis, and other recurring skin conditions—not a single diagnosis.
          </p>
          <p>
            <strong>Primary users:</strong> people logging flares, triggers, products, and treatment adherence.{" "}
            <strong>Secondary users:</strong> dermatologists and primary care clinicians who benefit from longitudinal
            context beyond recall alone.
          </p>
        </CardContent>
      </Card>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Symptom scales (0–10)</CardTitle>
          <CardDescription>Itch, pain, and stress sliders</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-foreground/80">
          <ul className="list-inside list-disc space-y-2">
            <li>
              <strong>Itch (0–10):</strong> 0 = none, 10 = worst imaginable. Use the same anchor each day when
              possible.
            </li>
            <li>
              <strong>Pain (0–10):</strong> burning, soreness, or tenderness from the affected skin—not other pain
              unless you choose to include it consistently.
            </li>
            <li>
              <strong>Stress (0–10):</strong> subjective stress level that day; useful for spotting correlations with
              flares, not a clinical stress instrument.
            </li>
            <li>
              <strong>Sleep (hours):</strong> approximate sleep the prior night; optional context for flares.
            </li>
          </ul>
        </CardContent>
      </Card>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Medications &amp; adherence</CardTitle>
          <CardDescription>What “remedy tracker” style fields mean</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-foreground/80">
          <p>
            <strong>Medications taken</strong> is free text for what you used that day (topical or systemic).{" "}
            <strong>Adherence</strong> means you took or applied medications as your plan intended—not that a specific
            dose was verified. This supports trend review; it does not replace pharmacy or medical records.
          </p>
        </CardContent>
      </Card>

      <Card className="glass-card border-slate-200/80 dark:border-slate-700">
        <CardHeader>
          <CardTitle className="text-lg">Insights &amp; analysis tab</CardTitle>
          <CardDescription>Trends and the optional image analysis widget</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-foreground/80">
          <p>
            <strong>Data analysis</strong> summarizes your saved symptom entries (averages and simple trend arrows). It
            reflects what you logged, not a clinical score like EASI or PASI unless you add those later.
          </p>
          <p>
            <strong>Image analysis</strong> in this web build uses a <strong>demonstration</strong> flow (simulated
            metrics). The Python/Streamlit prototype can compute real metrics (area, redness, border irregularity,
            asymmetry, ΔE) when that pipeline is ported or run server-side.
          </p>
        </CardContent>
      </Card>

      <Card className="glass-card border-amber-200/80 bg-[var(--st-warning-bg)] dark:border-amber-800 dark:bg-amber-950/30">
        <CardHeader>
          <CardTitle className="text-lg text-amber-950 dark:text-amber-100">Not a medical device</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-amber-950/90 dark:text-amber-100/90">
          <p className="mb-2">
            SkinTrack+ is <strong>not</strong> intended to diagnose, treat, or replace professional care. Metrics and
            charts are descriptive aids. Seek a qualified clinician for new, changing, or concerning lesions—especially
            pigmented or rapidly evolving spots.
          </p>
          <p>
            Data is stored <strong>on this device</strong> (browser storage and IndexedDB) unless you later enable
            cloud sync. Protect device access like any health notes.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
