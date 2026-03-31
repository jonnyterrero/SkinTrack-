import {
  EXPORT_SCHEMA_VERSION,
  emptyUserProfile,
  type SkinTrackExportV1,
  type SkinTrackRecord,
  type UserProfile,
} from "@/lib/types"
import type { DailyMedCheckoff, MedicationCatalogItem } from "@/lib/domain/medications"

export type ParseImportResult =
  | { ok: true; bundle: SkinTrackExportV1 }
  | { ok: false; error: string }

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v)
}

const LEGACY_VERSIONS = new Set([EXPORT_SCHEMA_VERSION, 1, "1", "1.0", "2", 2])

/**
 * Accepts v1/v2 export objects or legacy `{ version: "1.0", records, profile }`.
 */
export function parseSkinTrackImport(raw: unknown): ParseImportResult {
  if (!isPlainObject(raw)) {
    return { ok: false, error: "Invalid JSON: expected an object." }
  }

  const version = raw.version
  const records = raw.records
  if (!Array.isArray(records)) {
    return { ok: false, error: 'Missing or invalid "records" array.' }
  }

  if (!LEGACY_VERSIONS.has(version as never)) {
    return { ok: false, error: `Unsupported export version: ${String(version)}` }
  }

  const profile: UserProfile = isPlainObject(raw.profile)
    ? { ...emptyUserProfile(), ...(raw.profile as UserProfile) }
    : emptyUserProfile()

  const medicationCatalog = Array.isArray(raw.medicationCatalog)
    ? (raw.medicationCatalog as MedicationCatalogItem[])
    : undefined

  const medDailyByDate = isPlainObject(raw.medDailyByDate)
    ? (raw.medDailyByDate as Record<string, DailyMedCheckoff>)
    : undefined

  const bundle: SkinTrackExportV1 = {
    version: EXPORT_SCHEMA_VERSION,
    exportDate: typeof raw.exportDate === "string" ? raw.exportDate : new Date().toISOString(),
    records: records as SkinTrackRecord[],
    profile,
    medicationCatalog,
    medDailyByDate,
  }

  return { ok: true, bundle }
}

export function buildExportPayload(
  records: SkinTrackRecord[],
  profile: UserProfile,
  extras?: {
    medicationCatalog?: MedicationCatalogItem[]
    medDailyByDate?: Record<string, DailyMedCheckoff>
  },
): SkinTrackExportV1 {
  return {
    version: EXPORT_SCHEMA_VERSION,
    exportDate: new Date().toISOString(),
    records,
    profile,
    ...(extras?.medicationCatalog ? { medicationCatalog: extras.medicationCatalog } : {}),
    ...(extras?.medDailyByDate ? { medDailyByDate: extras.medDailyByDate } : {}),
  }
}
