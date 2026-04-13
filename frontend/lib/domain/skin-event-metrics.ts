/**
 * Locked v1 clinical metrics for skin events. Do not change ranges without bumping
 * METRICS_SCHEMA_VERSION and a migration plan.
 */
export const METRICS_SCHEMA_VERSION = 1 as const

/** Client lesion ids must stay RFC4122 so they map to Postgres `uuid` when syncing. */
export function newLesionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === "x" ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export const LESION_SEVERITY_0_4 = [
  { value: 0 as const, label: "Clear", description: "No visible lesion" },
  { value: 1 as const, label: "Mild", description: "Slight redness or irritation" },
  { value: 2 as const, label: "Moderate", description: "Visible lesion, discomfort" },
  { value: 3 as const, label: "Severe", description: "Inflamed, painful, or spreading" },
  { value: 4 as const, label: "Very severe", description: "Ulcerated, infected, or debilitating" },
] as const

export type LesionSeverity04 = (typeof LESION_SEVERITY_0_4)[number]["value"]

export const SYMPTOM_INTENSITY = {
  min: 0,
  max: 10,
  step: 1,
} as const

export const STRESS_SCALE = SYMPTOM_INTENSITY

export const SLEEP_HOURS_RANGE = {
  min: 0,
  max: 24,
  step: 0.5,
} as const

export const SLEEP_QUALITY = {
  min: 1,
  max: 5,
  step: 1,
  label: "Sleep quality",
} as const

/** Canonical location ids for v1 (single string per row; index-friendly for Postgres). */
export const SKIN_LOCATION_OPTIONS = [
  { id: "face.forehead", label: "Face — Forehead", group: "face" as const },
  { id: "face.cheeks", label: "Face — Cheeks", group: "face" as const },
  { id: "face.chin", label: "Face — Chin", group: "face" as const },
  { id: "face.other", label: "Face — Other", group: "face" as const },
  { id: "scalp", label: "Scalp", group: "coarse" as const },
  { id: "arms", label: "Arms", group: "coarse" as const },
  { id: "legs", label: "Legs", group: "coarse" as const },
  { id: "torso", label: "Torso", group: "coarse" as const },
  { id: "back", label: "Back", group: "coarse" as const },
] as const

export type SkinLocationId = (typeof SKIN_LOCATION_OPTIONS)[number]["id"]

export function isValidLocationId(id: string): id is SkinLocationId {
  return SKIN_LOCATION_OPTIONS.some((o) => o.id === id)
}

export function locationLabel(id: string): string {
  return SKIN_LOCATION_OPTIONS.find((o) => o.id === id)?.label ?? id
}

/** Map v1 location to coarse body-map circle id (see body-areas). */
export function locationIdToBodyMapAreaId(locationId: string): string | null {
  if (locationId.startsWith("face.")) return "head"
  if (locationId === "scalp") return "head"
  if (locationId === "arms") return "left-arm"
  if (locationId === "legs") return "left-thigh"
  if (locationId === "torso") return "chest"
  if (locationId === "back") return "chest"
  return null
}
