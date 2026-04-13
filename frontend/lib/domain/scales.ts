/** Canonical numeric scales used across symptom logging and analytics. */

export const SCALE_ZERO_TO_TEN = {
  min: 0,
  max: 10,
  step: 1,
  label: "0–10",
  description: "Higher means stronger intensity.",
} as const

export const SLEEP_HOURS = {
  min: 0,
  max: 24,
  step: 0.5,
  label: "Hours",
  description: "Approximate sleep duration last night.",
} as const

export const SYMPTOM_SCALE_FIELDS = ["itch", "pain", "stress"] as const
export type SymptomScaleField = (typeof SYMPTOM_SCALE_FIELDS)[number]

export const SYMPTOM_SCALE_LABELS: Record<SymptomScaleField, string> = {
  itch: "Itch",
  pain: "Pain",
  stress: "Stress",
}
