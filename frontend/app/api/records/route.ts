import { NextResponse, type NextRequest } from "next/server"
import { createRecordSchema } from "@/lib/validators/records"
import {
  requireAuthAndRateLimit,
  dbError,
  validationError,
} from "@/lib/api/helpers"

export async function GET(request: NextRequest) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { searchParams } = request.nextUrl
  const limit = Math.min(Number(searchParams.get("limit") || 100), 500)
  const offset = Number(searchParams.get("offset") || 0)
  const recordType = searchParams.get("type")

  let query = supabase
    .from("records")
    .select("*")
    .eq("user_id", user.id)
    .order("ts", { ascending: false })
    .range(offset, offset + limit - 1)

  if (recordType) {
    query = query.eq("record_type", recordType)
  }

  const { data, error } = await query
  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const body = await request.json()
  const parsed = createRecordSchema.safeParse(body)
  if (!parsed.success) return validationError(parsed.error)

  const { data, error } = await supabase
    .from("records")
    .insert({ ...parsed.data, user_id: user.id })
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
