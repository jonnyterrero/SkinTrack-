import { NextResponse } from "next/server"
import { updateSkinEventSchema } from "@/lib/validators/skin-events"
import {
  requireAuthAndRateLimit,
  notFound,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function GET(_request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params
  const { data, error } = await supabase
    .from("skin_events")
    .select("*")
    .eq("id", id)
    .eq("user_id", user.id)
    .single()

  if (error) {
    if (error.code === "PGRST116") return notFound("Skin event")
    return dbError(error.message)
  }
  return NextResponse.json(data)
}

export async function PATCH(request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params
  const result = await sanitizedBody(request, updateSkinEventSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("skin_events")
    .update(result.data)
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .single()

  if (error) {
    if (error.code === "PGRST116") return notFound("Skin event")
    return dbError(error.message)
  }
  return NextResponse.json(data)
}

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params
  const { error } = await supabase
    .from("skin_events")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id)

  if (error) return dbError(error.message)
  return new NextResponse(null, { status: 204 })
}
