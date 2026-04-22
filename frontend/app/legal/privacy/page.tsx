import Link from "next/link"

export const metadata = {
  title: "Privacy Policy · SkinTrack+",
  description: "How SkinTrack+ collects, uses, stores, and deletes your data.",
}

export default function PrivacyPolicy() {
  return (
    <article className="mx-auto max-w-2xl p-6 prose prose-slate">
      <Link href="/" className="text-sm underline text-slate-600">
        ← Back
      </Link>
      <h1>Privacy Policy</h1>
      <p>
        <em>Effective date: 2026-04-20</em>
      </p>

      <h2>Summary</h2>
      <ul>
        <li>You own your data. You can export or delete it at any time.</li>
        <li>We don&apos;t sell your data and we don&apos;t share it with advertisers.</li>
        <li>
          Your data is stored in your personal account in Supabase (Postgres +
          Storage). The app is also local-first: records and images are cached
          on your device so it works offline.
        </li>
        <li>We use cookies strictly for keeping you signed in.</li>
      </ul>

      <h2>What we collect</h2>
      <ul>
        <li>
          <strong>Account data:</strong> email address (for sign-in) and any
          display name you choose.
        </li>
        <li>
          <strong>Health-tracking data you enter:</strong> symptoms, severity
          scores, lesions, photos, medications, triggers, products, notes,
          body-map pins, reminders, and export history.
        </li>
        <li>
          <strong>Device / session cookies:</strong> set by Supabase Auth so
          you stay signed in.
        </li>
      </ul>

      <h2>What we do not collect</h2>
      <ul>
        <li>We do not use third-party advertising or analytics SDKs.</li>
        <li>We do not access your contacts, location, or microphone.</li>
        <li>
          Photos are captured only when you initiate a capture or upload. They
          are stored privately, readable only by you, via Supabase Storage with
          row-level security.
        </li>
      </ul>

      <h2>How we use your data</h2>
      <p>
        To provide the tracking features you request: storing records, syncing
        to the cloud for backup and cross-device use, generating summaries you
        choose to export, and sending you the reminders you enable.
      </p>

      <h2>Sharing</h2>
      <p>
        We do not share your data with third parties except:
      </p>
      <ul>
        <li>
          Our sub-processor Supabase, which hosts your encrypted data at rest.
        </li>
        <li>When you explicitly export or share a summary yourself.</li>
        <li>
          If required by law (subpoena, court order) — we will attempt to
          notify you where legally permitted.
        </li>
      </ul>

      <h2>Retention</h2>
      <p>
        Your data is kept until you delete it. You can delete specific records
        from the app, or delete your entire account from Settings → Danger
        zone, which removes all server-side rows and storage objects
        immediately.
      </p>

      <h2>Security</h2>
      <ul>
        <li>All traffic uses HTTPS/TLS.</li>
        <li>
          Row-level security policies in Postgres restrict each row to its
          owner.
        </li>
        <li>Images are stored in a private bucket with per-user RLS.</li>
        <li>API rate limiting is applied to prevent abuse.</li>
      </ul>

      <h2>Children</h2>
      <p>
        SkinTrack+ is not intended for children under 13 and we do not
        knowingly collect their data. Users between 13 and 18 should use the
        app only with a parent or guardian&apos;s permission.
      </p>

      <h2>Your rights</h2>
      <ul>
        <li>
          <strong>Access:</strong> Settings → Export my data returns a full
          JSON dump.
        </li>
        <li>
          <strong>Deletion:</strong> Settings → Danger zone → Delete my
          account.
        </li>
        <li>
          <strong>Portability:</strong> CSV, JSON, and printable PDF summary
          exports are built in.
        </li>
      </ul>

      <h2>Changes</h2>
      <p>
        We may update this policy. If we make material changes we will update
        the effective date and notify you in-app on your next sign-in.
      </p>

      <h2>Contact</h2>
      <p>
        For privacy questions, write to{" "}
        <a href="mailto:support@skintrack.app">support@skintrack.app</a>. See
        also the <Link href="/support">Support</Link> page.
      </p>
    </article>
  )
}
