import Link from "next/link"

export const metadata = {
  title: "Terms of Use · SkinTrack+",
  description: "Terms of Use for the SkinTrack+ application.",
}

export default function Terms() {
  return (
    <article className="mx-auto max-w-2xl p-6 prose prose-slate">
      <Link href="/" className="text-sm underline text-slate-600">
        ← Back
      </Link>
      <h1>Terms of Use</h1>
      <p>
        <em>Effective date: 2026-04-20</em>
      </p>

      <h2>1. Not a medical device</h2>
      <p>
        SkinTrack+ is a self-tracking application intended for personal use. It
        is <strong>not</strong> a medical device, it does <strong>not</strong>{" "}
        diagnose any condition, and it does <strong>not</strong> replace
        consultation with a licensed healthcare provider. Do not rely on
        SkinTrack+ for medical emergencies.
      </p>

      <h2>2. Eligibility</h2>
      <p>
        You must be at least 13 years old to use SkinTrack+. Users between 13
        and 18 should use it only with a parent or guardian&apos;s permission.
      </p>

      <h2>3. Your account and data</h2>
      <p>
        You are responsible for keeping your sign-in credentials safe. You are
        responsible for the content you enter. You retain ownership of your
        data and may export or delete it at any time.
      </p>

      <h2>4. Acceptable use</h2>
      <p>You agree not to:</p>
      <ul>
        <li>Use the service to harass or impersonate any other person.</li>
        <li>Attempt to circumvent security controls or rate limits.</li>
        <li>Upload malware or content that you do not have the right to share.</li>
        <li>Use the service for bulk scraping or automated misuse.</li>
      </ul>

      <h2>5. Availability</h2>
      <p>
        We aim for high availability but do not guarantee uninterrupted
        service. Scheduled maintenance and incidents may occur. Because the app
        is local-first, you will generally be able to continue logging data
        offline and it will sync when connectivity is restored.
      </p>

      <h2>6. Limitation of liability</h2>
      <p>
        To the maximum extent permitted by law, SkinTrack+ and its operators
        shall not be liable for any indirect, incidental, special, or
        consequential damages arising from the use of the service. The service
        is provided &quot;as is&quot; without warranty of any kind.
      </p>

      <h2>7. Changes</h2>
      <p>
        We may update these terms. Continued use after changes means you
        accept the updated terms. Material changes will be highlighted in-app.
      </p>

      <h2>8. Contact</h2>
      <p>
        Questions about these terms go to{" "}
        <a href="mailto:support@skintrack.app">support@skintrack.app</a>.
      </p>
    </article>
  )
}
