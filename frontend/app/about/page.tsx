import Link from "next/link"
import { Card } from "@/components/ui/card"

export const metadata = {
  title: "About SkinTrack+",
  description: "What SkinTrack+ does, what it doesn't do, and how to use it.",
}

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-2xl space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">About SkinTrack+</h1>
          <Link href="/" className="text-sm text-slate-600 underline">Back</Link>
        </div>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">What SkinTrack+ is</h2>
          <p className="text-sm text-slate-700">
            SkinTrack+ is a self-tracking app for people managing chronic skin
            conditions. It helps you log symptoms, capture photos, track lesions
            over time, and share a clean summary with your dermatologist.
          </p>
        </Card>

        <Card className="p-5 space-y-3 border-amber-200 bg-amber-50">
          <h2 className="text-lg font-medium text-amber-800">
            What SkinTrack+ is <em>not</em>
          </h2>
          <ul className="list-disc pl-5 text-sm text-amber-900 space-y-1">
            <li>Not a diagnostic tool. It does not diagnose skin conditions.</li>
            <li>Not a medical device and not reviewed as one by the FDA.</li>
            <li>
              Not a substitute for professional care. Always consult a licensed
              healthcare provider for medical advice.
            </li>
            <li>
              Not a substitute for emergency services. In an emergency, call
              your local emergency number.
            </li>
          </ul>
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Who uses it</h2>
          <div className="text-sm text-slate-700 space-y-2">
            <p>
              <strong>Patients</strong> use SkinTrack+ to notice patterns in
              their skin: what triggers flares, how a new medication is
              working, how a lesion looks over months instead of hours.
            </p>
            <p>
              <strong>Clinicians</strong> use the exported summaries to see
              longitudinal context between visits — symptom severity, adherence
              to prescribed treatment, photo progression, and suspected
              triggers.
            </p>
          </div>
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">How the symptom scales work</h2>
          <div className="text-sm text-slate-700 space-y-2">
            <div>
              <strong>Itch, pain, burning, dryness, stress</strong> — 0 (none)
              to 10 (worst you have ever experienced).
            </div>
            <div>
              <strong>Severity</strong> — 0 (clear) to 4 (very severe flare).
              Reflects overall condition at the time of logging.
            </div>
            <div>
              <strong>Sleep hours</strong> — total hours slept last night.
            </div>
            <div>
              <strong>Sleep quality</strong> — 1 (very poor) to 5 (excellent).
            </div>
          </div>
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">What the charts mean</h2>
          <div className="text-sm text-slate-700 space-y-2">
            <div>
              <strong>Area (cm²)</strong> — calibrated size of the lesion
              region. Calibration uses an ArUco marker or manual ruler entry;
              uncalibrated photos are shown in raw pixels and flagged.
            </div>
            <div>
              <strong>Redness index</strong> — relative red channel intensity
              inside the lesion mask vs. surrounding skin. Higher = redder.
            </div>
            <div>
              <strong>Border irregularity</strong> — how jagged the lesion edge
              is. Higher = more irregular.
            </div>
            <div>
              <strong>Asymmetry</strong> — how much the lesion deviates from a
              symmetric shape.
            </div>
            <div>
              <strong>Delta E</strong> — perceptual color distance between the
              lesion and nearby healthy skin (CIE Lab).
            </div>
          </div>
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Remedy tracking</h2>
          <p className="text-sm text-slate-700">
            &quot;Taken accordingly&quot; means you used the medication or remedy as
            prescribed or planned for that day. If you missed a dose, log it
            honestly — the goal is an accurate picture, not a perfect streak.
          </p>
        </Card>

        <Card className="p-5 space-y-3">
          <h2 className="text-lg font-medium">Privacy</h2>
          <p className="text-sm text-slate-700">
            Your data is stored in your account and on your device. We don&apos;t
            sell it and we don&apos;t share it with advertisers. See the{" "}
            <Link href="/legal/privacy" className="underline">
              Privacy Policy
            </Link>{" "}
            for details.
          </p>
        </Card>
      </div>
    </div>
  )
}
