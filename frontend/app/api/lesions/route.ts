import { NextResponse } from "next/server"
import { createLesionSchema } from "@/lib/validators/lesions"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { data, error } = await supabase
    .from("lesions")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })

  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, createLesionSchema)
  if (!result.ok) return result.response

  const row = {
    ...(result.data.id ? { id: result.data.id } : {}),
    user_id: user.id,
    label: result.data.label,
  }

  const { data, error } = await supabase
    .from("lesions")
    .insert(row)
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
