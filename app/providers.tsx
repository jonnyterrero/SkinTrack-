"use client"

import type { ReactNode } from "react"
import { Toaster } from "sonner"
import { SkinTrackProvider } from "@/components/skintrack-provider"

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <SkinTrackProvider>
      {children}
      <Toaster richColors position="top-center" closeButton />
    </SkinTrackProvider>
  )
}
