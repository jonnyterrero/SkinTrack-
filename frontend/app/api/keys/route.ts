import { NextResponse } from "next/server"
import { generateApiKey, hashApiKey } from "@/lib/api/api-keys"
import {
  requireAuthAndRateLimit,
  dbError,
} from "@/lib/api/helpers"

export async function GET() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { data, error } = await supabase
    .from("api_keys")
    .select("id, prefix, created_at, last_used_at")
    .eq("user_id", user.id)
    .is("revoked_at", null)
    .order("created_at", { ascending: false })

  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const raw = generateApiKey()
  const hashed = await hashApiKey(raw)
  const prefix = raw.slice(0, 10)

  const { data, error } = await supabase
    .from("api_keys")
    .insert({
      user_id: user.id,
      hash: hashed,
      prefix,
    })
    .select("id, prefix, created_at")
    .single()

  if (error) return dbError(error.message)

  // Return the raw key exactly once — clients must store it themselves.
  return NextResponse.json(
    { id: data.id, key: raw, prefix: data.prefix, created_at: data.created_at },
    { status: 201 },
  )
}
