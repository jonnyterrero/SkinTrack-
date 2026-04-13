import { NextResponse } from "next/server"
import {
  requireAuthAndRateLimit,
  notFound,
  dbError,
} from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params

  const { data, error } = await supabase
    .from("api_keys")
    .update({ revoked_at: new Date().toISOString() })
    .eq("id", id)
    .eq("user_id", user.id)
    .is("revoked_at", null)
    .select("id")
    .maybeSingle()

  if (error) return dbError(error.message)
  if (!data) return notFound("API key")

  return new NextResponse(null, { status: 204 })
}
