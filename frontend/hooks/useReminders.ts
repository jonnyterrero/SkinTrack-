"use client"

import { useEffect, useRef } from "react"
import { useAuth } from "@/context/AuthContext"
import { apiGet } from "@/lib/api/client"
import type { AppPreferences } from "@/lib/types/backend"

/**
 * Client-side reminder scheduler.
 *
 * Limitations to be honest about:
 * - Only fires while the PWA is open. True background reminders require a
 *   push service (web push + service worker on Android; Apple only recently
 *   enabled web push and only for installed PWAs).
 * - The UI surface (Settings) documents this limitation to the user.
 *
 * This hook is called from <AppShell> when the user is signed in.
 */
export function useReminders() {
  const { user } = useAuth()
  const timers = useRef<number[]>([])

  useEffect(() => {
    if (!user) return
    let cancelled = false

    async function wire() {
      try {
        const prefs = await apiGet<AppPreferences>("/api/app-preferences")
        if (cancelled) return
        if (!prefs.reminders_enabled) return
        if (typeof window === "undefined") return
        if (!("Notification" in window)) return
        if (Notification.permission === "default") {
          await Notification.requestPermission()
        }
        if (Notification.permission !== "granted") return

        const [h, m] = (prefs.preferred_log_time ?? "20:00:00").split(":").map(Number)
        const now = new Date()
        const next = new Date()
        next.setHours(h, m, 0, 0)
        if (next.getTime() <= now.getTime()) next.setDate(next.getDate() + 1)

        const delay = Math.max(5_000, next.getTime() - now.getTime())
        const id = window.setTimeout(() => {
          new Notification("SkinTrack+ daily log", {
            body: "Quick check-in: how is your skin today?",
            icon: "/icon-192x192.png",
            tag: "skintrack-daily-log",
          })
        }, delay)
        timers.current.push(id)
      } catch {
        // silent — reminders are best-effort
      }
    }

    wire()
    const timerIds = timers.current
    return () => {
      cancelled = true
      timerIds.forEach((id) => window.clearTimeout(id))
      timerIds.length = 0
    }
  }, [user])
}
