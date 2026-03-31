import type { SeverityLevel } from "@/lib/types"

export const SEVERITY_ORDER: SeverityLevel[] = ["low", "medium", "high"]

export const SEVERITY_DESCRIPTIONS: Record<SeverityLevel, string> = {
  low: "Mild symptoms; manageable day-to-day.",
  medium: "Noticeable impact on comfort or activity.",
  high: "Significant flare; consider contacting your clinician if persistent.",
}
