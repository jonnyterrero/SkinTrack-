"use client"

import { useState } from "react"
import { getSupabaseBrowserClient } from "@/lib/supabase/browser-client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function LoginPage() {
  const [email, setEmail] = useState("")
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle")
  const [errorMsg, setErrorMsg] = useState("")

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const supabase = getSupabaseBrowserClient()
    if (!supabase) {
      setStatus("error")
      setErrorMsg("Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.")
      return
    }

    setStatus("sending")
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
    })

    if (error) {
      setStatus("error")
      setErrorMsg(error.message)
    } else {
      setStatus("sent")
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Sign in to SkinTrack+</CardTitle>
          <CardDescription>
            Enter your email to receive a magic link.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {status === "sent" ? (
            <p className="text-sm text-muted-foreground">
              Check your email for a login link. You can close this tab.
            </p>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <Input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={status === "sending"}
              />
              <Button type="submit" className="w-full" disabled={status === "sending"}>
                {status === "sending" ? "Sending..." : "Send magic link"}
              </Button>
              {status === "error" && (
                <p className="text-sm text-destructive">{errorMsg}</p>
              )}
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
