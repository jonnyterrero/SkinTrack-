"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { getSupabaseBrowserClient } from "@/lib/supabase/browser-client"
import { useAuth } from "@/context/AuthContext"
import { runSyncCycle, type SyncState } from "@/lib/sync/engine"
import { queueSize } from "@/lib/sync/queue"

const SYNC_INTERVAL_MS = 30_000

export function useSyncEngine() {
  const { user } = useAuth()
  const [syncState, setSyncState] = useState<SyncState>("idle")
  const [pendingCount, setPendingCount] = useState(0)
  const [lastError, setLastError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const sync = useCallback(async () => {
    const supabase = getSupabaseBrowserClient()
    if (!supabase || !user) return

    setLastError(null)
    const result = await runSyncCycle(supabase, user.id, {
      onStateChange: setSyncState,
      onError: setLastError,
    })

    const remaining = await queueSize()
    setPendingCount(remaining)

    return result
  }, [user])

  useEffect(() => {
    if (!user) return

    void queueSize().then(setPendingCount)

    timerRef.current = setInterval(() => {
      void sync()
    }, SYNC_INTERVAL_MS)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [user, sync])

  return { syncState, pendingCount, lastError, sync }
}
