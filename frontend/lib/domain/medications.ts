export type MedScheduleSlot = "morning" | "afternoon" | "evening"

export type MedicationCatalogItem = {
  id: string
  name: string
  dose: string
  morning: boolean
  afternoon: boolean
  evening: boolean
  notes: string
}

export type MedAdherenceStatus = "taken" | "skipped" | "partial"

/** One calendar day: per-medication adherence from the catalog. */
export type DailyMedCheckoff = {
  date: string
  /** med id -> status */
  byMedicationId: Record<string, MedAdherenceStatus>
}

export function newMedicationId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `med_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
}

export function emptyMedicationItem(): MedicationCatalogItem {
  return {
    id: newMedicationId(),
    name: "",
    dose: "",
    morning: false,
    afternoon: false,
    evening: false,
    notes: "",
  }
}

export function defaultMedicationCatalog(): MedicationCatalogItem[] {
  return []
}
