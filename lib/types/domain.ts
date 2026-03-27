/** Canonical export bundle version (integer). */
export const EXPORT_SCHEMA_VERSION = 1 as const

export type UserProfile = {
  name: string
  age: string
  gender: string
  skinType: string
  conditions: string
  medications: string
  allergies: string
  notes: string
}

export const emptyUserProfile = (): UserProfile => ({
  name: "",
  age: "",
  gender: "",
  skinType: "",
  conditions: "",
  medications: "",
  allergies: "",
  notes: "",
})

export type SeverityLevel = "low" | "medium" | "high"

export type SymptomTrackRecord = {
  id: number
  timestamp: string
  type: "symptom"
  lesionLabel: string
  condition: string
  itch: number
  pain: number
  sleep: number
  stress: number
  triggers: string
  newProducts: string
  medications: string
  adherence: boolean
  notes: string
  bodyArea?: string
  severity?: SeverityLevel
}

/** Persisted image row: binary lives in IndexedDB via imageRef. */
export type ImageTrackRecord = {
  id: number
  timestamp: string
  type: "image"
  filename: string
  /** Stable id for IndexedDB blob */
  imageRef?: string
  /** Populated in memory after hydration (data URL) */
  image?: string
}

export type SkinTrackRecord = SymptomTrackRecord | ImageTrackRecord

/** Input when saving a new record (server assigns id + timestamp). */
export type NewSymptomRecordInput = Omit<SymptomTrackRecord, "id" | "timestamp">
export type NewImageRecordInput = {
  type: "image"
  filename: string
  /** data URL from file reader */
  image: string
}

export type NewSkinTrackRecordInput = NewSymptomRecordInput | NewImageRecordInput

/** Serialized row stored in localStorage (no inline image bytes). */
export type StoredSkinTrackRecord =
  | SymptomTrackRecord
  | Omit<ImageTrackRecord, "image"> & { imageRef: string }

/** Export file shape (version 1). Images embedded for portability. */
export type SkinTrackExportV1 = {
  version: typeof EXPORT_SCHEMA_VERSION
  exportDate: string
  records: SkinTrackRecord[]
  profile: UserProfile
}

export function isImageRecord(r: SkinTrackRecord): r is ImageTrackRecord {
  return r.type === "image"
}

export function isSymptomRecord(r: SkinTrackRecord): r is SymptomTrackRecord {
  return r.type === "symptom"
}

/** Row stored in localStorage for images (binary in IndexedDB). */
export type PersistedImageRow = {
  id: number
  timestamp: string
  type: "image"
  filename: string
  imageRef: string
}

export type PersistedRow = SymptomTrackRecord | PersistedImageRow
