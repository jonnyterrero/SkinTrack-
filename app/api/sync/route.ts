import { NextResponse } from "next/server"
import {
  requireAuthAndRateLimit,
  dbError,
  apiError,
} from "@/lib/api/helpers"

export async function POST(request: Request) {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const body = await request.json()
  if (!Array.isArray(body.operations)) {
    return apiError("VALIDATION_ERROR", "Expected { operations: [...] }", 400)
  }

  const results: Array<{ row_id: string; ok: boolean; error?: string }> = []

  for (const op of body.operations as Array<{
    table: string
    action: string
    row_id: string
    payload: Record<string, unknown>
  }>) {
    const { table, action, row_id, payload } = op

    if (!["records", "lesions", "skin_events", "profiles"].includes(table)) {
      results.push({ row_id, ok: false, error: `Invalid table: ${table}` })
      continue
    }

    if (action === "delete") {
      const { error } = await supabase
        .from(table)
        .delete()
        .eq("id", row_id)
        .eq(table === "profiles" ? "id" : "user_id", user.id)
      results.push({ row_id, ok: !error, error: error?.message })
      continue
    }

    if (action === "upsert") {
      const row = {
        ...payload,
        [table === "profiles" ? "id" : "user_id"]: user.id,
      }
      if (row_id && table !== "profiles") {
        ;(row as Record<string, unknown>).id = row_id
      }
      const { error } = await supabase
        .from(table)
        .upsert(row, { onConflict: "id" })
      results.push({ row_id, ok: !error, error: error?.message })
      continue
    }

    results.push({ row_id, ok: false, error: `Invalid action: ${action}` })
  }

  const allOk = results.every((r) => r.ok)
  if (!allOk) {
    const firstErr = results.find((r) => !r.ok)
    return dbError(firstErr?.error ?? "Partial sync failure")
  }

  return NextResponse.json({ ok: true, processed: results.length })
}
