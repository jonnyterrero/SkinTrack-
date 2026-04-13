"use client"

import type { ReactNode } from "react"
import { Toaster } from "sonner"
import { AuthProvider } from "@/context/AuthContext"
import { SkinTrackProvider } from "@/components/skintrack-provider"

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <SkinTrackProvider>
        {children}
        <Toaster richColors position="top-center" closeButton />
      </SkinTrackProvider>
    </AuthProvider>
  )
}
