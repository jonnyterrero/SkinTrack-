import type { SupabaseClient } from "@supabase/supabase-js"
import { peek, dequeue, incrementRetry, queueSize, type SyncOperation } from "./queue"

export type SyncState = "idle" | "dirty" | "syncing" | "synced" | "error"

const MAX_RETRIES = 5

export type SyncEngineCallbacks = {
  onStateChange?: (state: SyncState) => void
  onError?: (message: string) => void
  onProgress?: (completed: number, total: number) => void
}

async function processOperation(
  supabase: SupabaseClient,
  userId: string,
  op: SyncOperation,
): Promise<{ ok: boolean; error?: string }> {
  const { table, action, row_id, payload } = op

  if (action === "delete") {
    const { error } = await supabase
      .from(table)
      .delete()
      .eq("id", row_id)
      .eq(table === "profiles" ? "id" : "user_id", userId)

    return error ? { ok: false, error: error.message } : { ok: true }
  }

  // upsert
  const row = { ...payload, [table === "profiles" ? "id" : "user_id"]: userId }
  if (row_id && table !== "profiles") {
    ;(row as Record<string, unknown>).id = row_id
  }

  const { error } = await supabase
    .from(table)
    .upsert(row, { onConflict: "id" })

  return error ? { ok: false, error: error.message } : { ok: true }
}

export async function runSyncCycle(
  supabase: SupabaseClient,
  userId: string,
  callbacks?: SyncEngineCallbacks,
): Promise<SyncState> {
  const total = await queueSize()
  if (total === 0) {
    callbacks?.onStateChange?.("synced")
    return "synced"
  }

  callbacks?.onStateChange?.("syncing")
  let completed = 0

  const batch = await peek(50)
  for (const op of batch) {
    if (op.retries >= MAX_RETRIES) {
      if (op.id != null) await dequeue(op.id)
      continue
    }

    const result = await processOperation(supabase, userId, op)
    if (result.ok) {
      if (op.id != null) await dequeue(op.id)
      completed++
      callbacks?.onProgress?.(completed, total)
    } else {
      if (op.id != null) await incrementRetry(op.id)
      callbacks?.onError?.(result.error ?? "Unknown sync error")
      callbacks?.onStateChange?.("error")
      return "error"
    }
  }

  const remaining = await queueSize()
  const finalState: SyncState = remaining > 0 ? "dirty" : "synced"
  callbacks?.onStateChange?.(finalState)
  return finalState
}
