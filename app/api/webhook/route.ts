import { NextResponse, type NextRequest } from "next/server"
import {
  getAuthenticatedClient,
  unauthorized,
  apiError,
} from "@/lib/api/helpers"

export async function POST(request: NextRequest) {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const body = await request.json()
  const { event, payload } = body as { event?: string; payload?: unknown }

  if (!event) {
    return apiError("VALIDATION_ERROR", "Missing 'event' field.", 400)
  }

  const { data: profile } = await supabase
    .from("profiles")
    .select("skintrack_profile")
    .eq("id", user.id)
    .single()

  const webhookUrl = (profile?.skintrack_profile as Record<string, unknown>)?.webhookUrl
  if (typeof webhookUrl !== "string" || !webhookUrl) {
    return apiError("VALIDATION_ERROR", "No webhook URL configured.", 400)
  }

  try {
    const res = await fetch(webhookUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, payload, user_id: user.id, ts: new Date().toISOString() }),
    })

    return NextResponse.json({
      ok: res.ok,
      status: res.status,
    })
  } catch (err) {
    return apiError("DB_ERROR", (err as Error).message, 502)
  }
}
