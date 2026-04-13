import { NextResponse } from "next/server"
import { generateApiKey, hashApiKey } from "@/lib/api/api-keys"
import {
  getAuthenticatedClient,
  unauthorized,
  dbError,
} from "@/lib/api/helpers"

export async function POST() {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) return unauthorized()

  const raw = generateApiKey()
  const hashed = await hashApiKey(raw)

  const { data: profile, error: fetchErr } = await supabase
    .from("profiles")
    .select("skintrack_profile")
    .eq("id", user.id)
    .single()

  if (fetchErr) return dbError(fetchErr.message)

  const existing = (profile?.skintrack_profile ?? {}) as Record<string, unknown>
  const keys = Array.isArray(existing.apiKeys) ? existing.apiKeys : []
  keys.push({
    id: crypto.randomUUID(),
    hash: hashed,
    prefix: raw.slice(0, 10),
    created_at: new Date().toISOString(),
  })

  const { error: updateErr } = await supabase
    .from("profiles")
    .update({ skintrack_profile: { ...existing, apiKeys: keys } })
    .eq("id", user.id)

  if (updateErr) return dbError(updateErr.message)

  return NextResponse.json({ key: raw, prefix: raw.slice(0, 10) }, { status: 201 })
}
