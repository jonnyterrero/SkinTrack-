import { NextResponse } from "next/server"
import { createLesionSchema } from "@/lib/validators/lesions"
import {
  requireAuthAndRateLimit,
  dbError,
  validationError,
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

  const body = await request.json()
  const parsed = createLesionSchema.safeParse(body)
  if (!parsed.success) return validationError(parsed.error)

  const row = {
    ...(parsed.data.id ? { id: parsed.data.id } : {}),
    user_id: user.id,
    label: parsed.data.label,
  }

  const { data, error } = await supabase
    .from("lesions")
    .insert(row)
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
