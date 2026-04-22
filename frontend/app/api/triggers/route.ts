import { NextResponse } from "next/server"
import { requireAuthAndRateLimit, dbError } from "@/lib/api/helpers"

export async function GET() {
  const { supabase, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { data, error } = await supabase
    .from("trigger_taxonomy")
    .select("key, label, value_type")
    .order("label", { ascending: true })

  if (error) return dbError(error.message)
  return NextResponse.json(data ?? [])
}
