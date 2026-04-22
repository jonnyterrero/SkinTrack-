import { NextResponse, type NextRequest } from "next/server"
import { createLesionLocationSchema } from "@/lib/validators/lesion-locations"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET(request: NextRequest) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { searchParams } = request.nextUrl
  const lesionId = searchParams.get("lesion_id")

  let query = supabase
    .from("lesion_locations")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })

  if (lesionId) query = query.eq("lesion_id", lesionId)

  const { data, error } = await query
  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, createLesionLocationSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("lesion_locations")
    .insert({ ...result.data, user_id: user.id })
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
