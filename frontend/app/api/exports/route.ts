import { NextResponse } from "next/server"
import { createExportSchema } from "@/lib/validators/event-assets"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { data, error } = await supabase
    .from("exports")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .limit(50)

  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, createExportSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("exports")
    .insert({ ...result.data, user_id: user.id })
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
