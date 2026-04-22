import { NextResponse } from "next/server"
import {
  requireAuthAndRateLimit,
  dbError,
  notFound,
  sanitizedBody,
} from "@/lib/api/helpers"
import { updateUserAllergySchema } from "@/lib/validators/user-allergies"

export async function PATCH(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await context.params
  const result = await sanitizedBody(request, updateUserAllergySchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("user_allergies")
    .update(result.data)
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .maybeSingle()

  if (error) return dbError(error.message)
  if (!data) return notFound("Allergy")
  return NextResponse.json(data)
}

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await context.params
  const { data, error } = await supabase
    .from("user_allergies")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .maybeSingle()

  if (error) return dbError(error.message)
  if (!data) return notFound("Allergy")
  return NextResponse.json({ ok: true })
}
