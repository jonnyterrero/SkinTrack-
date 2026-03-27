"use client"

import type { ReactNode } from "react"
import { SkinTrackProvider } from "@/components/skintrack-provider"

export function AppProviders({ children }: { children: ReactNode }) {
  return <SkinTrackProvider>{children}</SkinTrackProvider>
}
