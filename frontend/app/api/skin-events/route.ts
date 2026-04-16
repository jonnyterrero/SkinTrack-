import { NextResponse, type NextRequest } from "next/server"
import { createSkinEventSchema } from "@/lib/validators/skin-events"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET(request: NextRequest) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { searchParams } = request.nextUrl
  const limit = Math.min(Number(searchParams.get("limit") || 100), 500)
  const offset = Number(searchParams.get("offset") || 0)
  const lesionId = searchParams.get("lesion_id")

  let query = supabase
    .from("skin_events")
    .select("*")
    .eq("user_id", user.id)
    .order("ts", { ascending: false })
    .range(offset, offset + limit - 1)

  if (lesionId) {
    query = query.eq("lesion_id", lesionId)
  }

  const { data, error } = await query
  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, createSkinEventSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("skin_events")
    .insert({ ...result.data, user_id: user.id })
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
