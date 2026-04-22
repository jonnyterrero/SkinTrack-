import { NextResponse } from "next/server"
import { updateAppPreferencesSchema } from "@/lib/validators/app-preferences"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { data, error } = await supabase
    .from("app_preferences")
    .select("*")
    .eq("user_id", user.id)
    .maybeSingle()

  if (error) return dbError(error.message)

  if (!data) {
    const { data: created, error: insertError } = await supabase
      .from("app_preferences")
      .insert({ user_id: user.id })
      .select()
      .single()
    if (insertError) return dbError(insertError.message)
    return NextResponse.json(created)
  }
  return NextResponse.json(data)
}

export async function PUT(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, updateAppPreferencesSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("app_preferences")
    .upsert({ user_id: user.id, ...result.data }, { onConflict: "user_id" })
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data)
}
