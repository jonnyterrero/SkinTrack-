import { NextResponse } from "next/server"
import { createUserConditionSchema } from "@/lib/validators/user-conditions"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { data, error } = await supabase
    .from("user_conditions")
    .select("*, conditions(id, slug, name, display_name, supports_imaging, supports_body_map)")
    .eq("user_id", user.id)
    .order("created_at", { ascending: true })

  if (error) return dbError(error.message)
  return NextResponse.json(data ?? [])
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, createUserConditionSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("user_conditions")
    .upsert(
      { user_id: user.id, ...result.data },
      { onConflict: "user_id,condition_id" },
    )
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
