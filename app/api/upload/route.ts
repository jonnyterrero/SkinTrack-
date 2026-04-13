import { NextResponse, type NextRequest } from "next/server"
import {
  MAX_FILE_SIZE,
  validateMimeType,
  validateMagicBytes,
} from "@/lib/validators/upload"
import {
  requireAuthAndRateLimit,
  apiError,
} from "@/lib/api/helpers"

export async function POST(request: NextRequest) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const contentType = request.headers.get("content-type") ?? ""
  if (!contentType.includes("multipart/form-data")) {
    return apiError("VALIDATION_ERROR", "Expected multipart/form-data.", 400)
  }

  const formData = await request.formData()
  const file = formData.get("file") as File | null
  const recordId = formData.get("record_id") as string | null
  const filename = formData.get("filename") as string | null

  if (!file) {
    return apiError("VALIDATION_ERROR", "File is required.", 400)
  }

  if (file.size > MAX_FILE_SIZE) {
    return apiError("VALIDATION_ERROR", `File exceeds ${MAX_FILE_SIZE / 1024 / 1024}MB limit.`, 400)
  }

  if (!validateMimeType(file.type)) {
    return apiError("VALIDATION_ERROR", `Unsupported file type: ${file.type}`, 400)
  }

  const buffer = await file.arrayBuffer()
  if (!validateMagicBytes(buffer, file.type)) {
    return apiError("VALIDATION_ERROR", "File content does not match declared type.", 400)
  }

  const safeName = (filename || file.name).replace(/[^a-zA-Z0-9._-]/g, "_")
  const folder = recordId || crypto.randomUUID()
  const storagePath = `${user.id}/${folder}/${safeName}`

  const { error: uploadError } = await supabase.storage
    .from("skintrack-images")
    .upload(storagePath, buffer, {
      contentType: file.type,
      upsert: true,
    })

  if (uploadError) {
    return apiError("STORAGE_ERROR", uploadError.message, 500)
  }

  return NextResponse.json({ path: storagePath }, { status: 201 })
}

export async function GET(request: NextRequest) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const path = request.nextUrl.searchParams.get("path")
  if (!path) {
    return apiError("VALIDATION_ERROR", "Query param 'path' is required.", 400)
  }

  if (!path.startsWith(`${user.id}/`)) {
    return apiError("FORBIDDEN", "Access denied.", 403)
  }

  const { data, error } = await supabase.storage
    .from("skintrack-images")
    .createSignedUrl(path, 3600)

  if (error) {
    return apiError("STORAGE_ERROR", error.message, 500)
  }

  return NextResponse.json({ signedUrl: data.signedUrl })
}
