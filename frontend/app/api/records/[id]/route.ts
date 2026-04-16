import { NextResponse } from "next/server"
import { updateRecordSchema } from "@/lib/validators/records"
import {
  requireAuthAndRateLimit,
  notFound,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function PATCH(request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params
  const result = await sanitizedBody(request, updateRecordSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("records")
    .update(result.data)
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .single()

  if (error) {
    if (error.code === "PGRST116") return notFound("Record")
    return dbError(error.message)
  }
  return NextResponse.json(data)
}

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params
  const { error } = await supabase
    .from("records")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id)

  if (error) return dbError(error.message)
  return new NextResponse(null, { status: 204 })
}
