import { NextResponse, type NextRequest } from "next/server"
import { createMedicationSchema } from "@/lib/validators/medications"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET(request: NextRequest) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { searchParams } = request.nextUrl
  const activeOnly = searchParams.get("active") === "true"

  let query = supabase
    .from("med_catalog")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })

  if (activeOnly) query = query.eq("active", true)

  const { data, error } = await query
  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, createMedicationSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("med_catalog")
    .insert({ ...result.data, user_id: user.id })
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
