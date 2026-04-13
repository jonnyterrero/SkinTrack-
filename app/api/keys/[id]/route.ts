import { NextResponse } from "next/server"
import {
  getAuthenticatedClient,
  unauthorized,
  notFound,
  dbError,
} from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const { id } = await params

  const { data: profile, error: fetchErr } = await supabase
    .from("profiles")
    .select("skintrack_profile")
    .eq("id", user.id)
    .single()

  if (fetchErr) return dbError(fetchErr.message)

  const existing = (profile?.skintrack_profile ?? {}) as Record<string, unknown>
  const keys = Array.isArray(existing.apiKeys) ? existing.apiKeys : []
  const filtered = keys.filter(
    (k: Record<string, unknown>) => k.id !== id,
  )

  if (filtered.length === keys.length) {
    return notFound("API key")
  }

  const { error: updateErr } = await supabase
    .from("profiles")
    .update({ skintrack_profile: { ...existing, apiKeys: filtered } })
    .eq("id", user.id)

  if (updateErr) return dbError(updateErr.message)

  return new NextResponse(null, { status: 204 })
}
