"use client"

import { AlertTriangle, X } from "lucide-react"
import { Button } from "@/components/ui/button"

type Props = {
  message: string | null
  onDismiss: () => void
}

export function StorageErrorBanner({ message, onDismiss }: Props) {
  if (!message) return null

  return (
    <div
      role="alert"
      className="mb-4 flex items-start gap-3 rounded-xl border border-amber-200/90 bg-[var(--st-warning-bg)] p-4 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100"
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="font-semibold">Storage issue</p>
        <p className="mt-1 text-amber-900/90 dark:text-amber-100/90">{message}</p>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="shrink-0 text-amber-900 hover:bg-amber-200/50 dark:text-amber-100 dark:hover:bg-amber-900/50"
        onClick={onDismiss}
        aria-label="Dismiss storage warning"
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  )
}
