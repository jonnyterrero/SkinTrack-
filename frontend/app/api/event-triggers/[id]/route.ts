import { NextResponse } from "next/server"
import { requireAuthAndRateLimit, dbError } from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params
  const { error } = await supabase
    .from("event_triggers")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id)

  if (error) return dbError(error.message)
  return new NextResponse(null, { status: 204 })
}
