import Link from "next/link"

export const metadata = {
  title: "Support · SkinTrack+",
  description: "How to get help using SkinTrack+.",
}

export default function Support() {
  return (
    <article className="mx-auto max-w-2xl p-6 prose prose-slate">
      <Link href="/" className="text-sm underline text-slate-600">
        ← Back
      </Link>
      <h1>Support</h1>

      <h2>Contact</h2>
      <p>
        Email us at{" "}
        <a href="mailto:support@skintrack.app">support@skintrack.app</a>. We
        aim to reply within two business days.
      </p>

      <h2>Common questions</h2>

      <h3>How do I delete my account?</h3>
      <p>
        Settings → Danger zone → Delete my account. This permanently removes
        all records, images, medications, and preferences.
      </p>

      <h3>How do I export my data?</h3>
      <p>
        Settings → Your data → Export my data gives you a full JSON dump. For
        CSV or a clinician summary, open the Export center.
      </p>

      <h3>The reminder didn&apos;t arrive.</h3>
      <p>
        Reminders currently fire while the app is open. Reliable background
        reminders require push notifications, which we plan to add next. In
        the meantime, keep the app installed on your home screen so it resumes
        quickly.
      </p>

      <h3>Is my data private?</h3>
      <p>
        Yes. Row-level security in Postgres makes every row readable only by
        its owner. Photos are stored in a private bucket keyed to your user
        id. See our{" "}
        <Link href="/legal/privacy">Privacy Policy</Link> for details.
      </p>

      <h3>SkinTrack+ isn&apos;t a medical device, right?</h3>
      <p>
        Correct. It&apos;s a self-tracking tool. It doesn&apos;t diagnose and it
        doesn&apos;t replace professional medical advice. See the{" "}
        <Link href="/about">About</Link> page for the full statement.
      </p>
    </article>
  )
}
