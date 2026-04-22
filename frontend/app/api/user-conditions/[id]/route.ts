import { NextResponse } from "next/server"
import { requireAuthAndRateLimit, dbError, notFound } from "@/lib/api/helpers"

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await context.params
  const { data, error } = await supabase
    .from("user_conditions")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .maybeSingle()

  if (error) return dbError(error.message)
  if (!data) return notFound("User condition")
  return NextResponse.json({ ok: true })
}
