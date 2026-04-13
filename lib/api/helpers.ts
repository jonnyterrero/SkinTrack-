import { NextResponse } from "next/server"
import type { ZodError } from "zod"
import { getSupabaseServerClient } from "@/lib/supabase/server"

export type ApiErrorCode =
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "VALIDATION_ERROR"
  | "DB_ERROR"
  | "STORAGE_ERROR"
  | "CONFLICT"
  | "NOT_FOUND"

export function apiError(
  code: ApiErrorCode,
  message: string,
  status: number,
  details?: unknown,
) {
  return NextResponse.json({ error: message, code, details }, { status })
}

export function validationError(err: ZodError) {
  return apiError(
    "VALIDATION_ERROR",
    "Invalid request body.",
    400,
    err.flatten().fieldErrors,
  )
}

export function unauthorized() {
  return apiError("UNAUTHORIZED", "Sign in required.", 401)
}

export function notFound(resource = "Resource") {
  return apiError("NOT_FOUND", `${resource} not found.`, 404)
}

export function dbError(message: string) {
  return apiError("DB_ERROR", message, 500)
}

export async function getAuthenticatedClient() {
  const supabase = await getSupabaseServerClient()
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser()
  if (error || !user) return { supabase: null, user: null }
  return { supabase, user }
}
