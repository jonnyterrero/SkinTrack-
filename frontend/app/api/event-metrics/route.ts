import { NextResponse, type NextRequest } from "next/server"
import { upsertEventMetricsSchema } from "@/lib/validators/event-assets"
import {
  requireAuthAndRateLimit,
  dbError,
  sanitizedBody,
} from "@/lib/api/helpers"

export async function GET(request: NextRequest) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { searchParams } = request.nextUrl
  const eventId = searchParams.get("skin_event_id")

  let query = supabase
    .from("event_metrics")
    .select("*")
    .eq("user_id", user.id)
    .order("id", { ascending: false })

  if (eventId) query = query.eq("skin_event_id", eventId)

  const { data, error } = await query
  if (error) return dbError(error.message)
  return NextResponse.json(data)
}

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const result = await sanitizedBody(request, upsertEventMetricsSchema)
  if (!result.ok) return result.response

  const { data, error } = await supabase
    .from("event_metrics")
    .upsert(
      { ...result.data, user_id: user.id },
      { onConflict: "skin_event_id" },
    )
    .select()
    .single()

  if (error) return dbError(error.message)
  return NextResponse.json(data, { status: 201 })
}
