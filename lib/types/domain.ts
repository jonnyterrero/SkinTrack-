import type { DailyMedCheckoff, MedicationCatalogItem } from "@/lib/domain/medications"
import { METRICS_SCHEMA_VERSION } from "@/lib/domain/skin-event-metrics"

/** Canonical export bundle version (integer). Bump when bundle shape changes. */
export const EXPORT_SCHEMA_VERSION = 3 as const

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
  medicationIds?: string[]
}

/** Local lesion registry (client-generated ids until hybrid sync). */
export type Lesion = {
  id: string
  label: string
  createdAt: string
  archived?: boolean
}

/** v1 locked metrics — longitudinal skin event log row. */
export type SkinEventRecord = {
  id: number
  timestamp: string
  type: "skin_event"
  lesionId: string
  severity04: 0 | 1 | 2 | 3 | 4
  locationId: string
  itch: number
  pain: number
  burning: number
  dryness: number
  stress: number
  sleepHours: number
  sleepQuality: 1 | 2 | 3 | 4 | 5
  metricsSchemaVersion: typeof METRICS_SCHEMA_VERSION
  notes?: string
}

/** Extra fields stored with image blobs (EXIF-light, location context). */
export type ImageMetadata = {
  capturedAt?: string
  bodyArea?: string
  source?: "camera" | "upload"
  note?: string
  width?: number
  height?: number
}

/** Persisted image row: binary lives in IndexedDB via imageRef. */
export type ImageTrackRecord = {
  id: number
  timestamp: string
  type: "image"
  filename: string
  imageRef?: string
  image?: string
  metadata?: ImageMetadata
}

export type SkinTrackRecord = SymptomTrackRecord | ImageTrackRecord | SkinEventRecord

/** Input when saving a new record (server assigns id + timestamp). */
export type NewSymptomRecordInput = Omit<SymptomTrackRecord, "id" | "timestamp">
export type NewImageRecordInput = {
  type: "image"
  filename: string
  image: string
  metadata?: ImageMetadata
}

export type NewSkinEventInput = Omit<SkinEventRecord, "id" | "timestamp" | "metricsSchemaVersion">

export type NewSkinTrackRecordInput = NewSymptomRecordInput | NewImageRecordInput | NewSkinEventInput

/** Serialized row stored in localStorage (no inline image bytes). */
export type StoredSkinTrackRecord =
  | SymptomTrackRecord
  | SkinEventRecord
  | Omit<ImageTrackRecord, "image"> & { imageRef: string }

/** Export file shape (v3 adds lesions registry). */
export type SkinTrackExportV1 = {
  version: typeof EXPORT_SCHEMA_VERSION
  exportDate: string
  records: SkinTrackRecord[]
  profile: UserProfile
  medicationCatalog?: MedicationCatalogItem[]
  medDailyByDate?: Record<string, DailyMedCheckoff>
  lesions?: Lesion[]
}

export function isImageRecord(r: SkinTrackRecord): r is ImageTrackRecord {
  return r.type === "image"
}

export function isSymptomRecord(r: SkinTrackRecord): r is SymptomTrackRecord {
  return r.type === "symptom"
}

export function isSkinEventRecord(r: SkinTrackRecord): r is SkinEventRecord {
  return r.type === "skin_event"
}

/** Row stored in localStorage for images (binary in IndexedDB). */
export type PersistedImageRow = {
  id: number
  timestamp: string
  type: "image"
  filename: string
  imageRef: string
  metadata?: ImageMetadata
}

export type PersistedRow = SymptomTrackRecord | PersistedImageRow | SkinEventRecord
