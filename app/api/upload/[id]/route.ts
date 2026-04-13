import { NextResponse } from "next/server"
import {
  requireAuthAndRateLimit,
  apiError,
} from "@/lib/api/helpers"

type Params = { params: Promise<{ id: string }> }

export async function DELETE(_request: Request, { params }: Params) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const { id: recordId } = await params

  const { data: files, error: listError } = await supabase.storage
    .from("skintrack-images")
    .list(`${user.id}/${recordId}`)

  if (listError) {
    return apiError("STORAGE_ERROR", listError.message, 500)
  }

  if (!files || files.length === 0) {
    return new NextResponse(null, { status: 204 })
  }

  const paths = files.map((f) => `${user.id}/${recordId}/${f.name}`)
  const { error: removeError } = await supabase.storage
    .from("skintrack-images")
    .remove(paths)

  if (removeError) {
    return apiError("STORAGE_ERROR", removeError.message, 500)
  }

  return new NextResponse(null, { status: 204 })
}
