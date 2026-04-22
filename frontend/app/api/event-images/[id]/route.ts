import { NextResponse } from "next/server"
import { requireAuthAndRateLimit, dbError } from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id } = await params
  const { data: row, error: fetchErr } = await supabase
    .from("event_images")
    .select("storage_path")
    .eq("id", id)
    .eq("user_id", user.id)
    .maybeSingle()
  if (fetchErr) return dbError(fetchErr.message)
  if (!row) return new NextResponse(null, { status: 204 })

  if (row.storage_path) {
    await supabase.storage.from("skintrack-images").remove([row.storage_path])
  }

  const { error } = await supabase
    .from("event_images")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id)
  if (error) return dbError(error.message)
  return new NextResponse(null, { status: 204 })
}
