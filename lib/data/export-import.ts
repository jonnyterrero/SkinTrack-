import {
  EXPORT_SCHEMA_VERSION,
  emptyUserProfile,
  type SkinTrackExportV1,
  type SkinTrackRecord,
  type UserProfile,
} from "@/lib/types"

export type ParseImportResult =
  | { ok: true; bundle: SkinTrackExportV1 }
  | { ok: false; error: string }

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v)
}

/**
 * Accepts v1 export objects or legacy `{ version: "1.0", records, profile }`.
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

  const v =
    version === EXPORT_SCHEMA_VERSION ||
    version === 1 ||
    version === "1" ||
    version === "1.0"
      ? EXPORT_SCHEMA_VERSION
      : null

  if (v === null) {
    return { ok: false, error: `Unsupported export version: ${String(version)}` }
  }

  const profile: UserProfile = isPlainObject(raw.profile)
    ? { ...emptyUserProfile(), ...(raw.profile as UserProfile) }
    : emptyUserProfile()

  const bundle: SkinTrackExportV1 = {
    version: EXPORT_SCHEMA_VERSION,
    exportDate: typeof raw.exportDate === "string" ? raw.exportDate : new Date().toISOString(),
    records: records as SkinTrackRecord[],
    profile,
  }

  return { ok: true, bundle }
}

export function buildExportPayload(
  records: SkinTrackRecord[],
  profile: UserProfile,
): SkinTrackExportV1 {
  return {
    version: EXPORT_SCHEMA_VERSION,
    exportDate: new Date().toISOString(),
    records,
    profile,
  }
}
