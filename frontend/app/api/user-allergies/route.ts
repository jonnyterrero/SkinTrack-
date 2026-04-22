import { NextResponse } from "next/server"
import { createUserAllergySchema } from "@/lib/validators/user-allergies"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { data, error } = await supabase
    .from("user_allergies")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: true })

  if (error) return dbError(error.message)
  return NextResponse.json(data ?? [])
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, createUserAllergySchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("user_allergies")
    .insert({ user_id: user.id, ...result.data })
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
