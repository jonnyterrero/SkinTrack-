import { NextResponse } from "next/server"
import type { ZodError } from "zod"
import { getSupabaseServerClient } from "@/lib/supabase/server"
import { checkRateLimit } from "./rate-limit"

export type ApiErrorCode =
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "VALIDATION_ERROR"
  | "DB_ERROR"
  | "STORAGE_ERROR"
  | "CONFLICT"
  | "NOT_FOUND"
  | "RATE_LIMITED"

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

/**
 * Auth + rate-limit guard for API routes.
 *
 * Pattern:
 *   const { supabase, user, error } = await requireAuthAndRateLimit()
 *   if (error) return error
 *
 * Returns a 401 if not signed in, a 429 if the user is over the
 * 100-req/60-s budget, or `{ supabase, user, error: null }` on success.
 */
export async function requireAuthAndRateLimit() {
  const { supabase, user } = await getAuthenticatedClient()
  if (!supabase || !user) {
    return { supabase: null, user: null, error: unauthorized() }
  }

  const allowed = await checkRateLimit(supabase, user.id)
  if (!allowed) {
    return {
      supabase: null,
      user: null,
      error: apiError(
        "RATE_LIMITED",
        "Rate limit exceeded. Try again in 60 seconds.",
        429,
      ),
    }
  }

  return { supabase, user, error: null as null }
}
