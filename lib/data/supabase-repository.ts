import { buildExportPayload, parseSkinTrackImport } from "@/lib/data/export-import"
import type { SkinTrackRepository, ImportBundleResult, PersistResult, SaveRecordResult } from "@/lib/data/repository"
import type { DailyMedCheckoff, MedicationCatalogItem } from "@/lib/domain/medications"
import { defaultMedicationCatalog } from "@/lib/domain/medications"
import {
  emptyUserProfile,
  type Lesion,
  type NewSkinTrackRecordInput,
  type SkinTrackExportV1,
  type SkinTrackRecord,
  type UserProfile,
} from "@/lib/types"

const notConfigured = (): never => {
  throw new Error(
    "Use the local repository for reads/writes. Cloud sync runs via syncLocalBundleToSupabase after sign-in.",
  )
}

/**
 * Stub for a future remote-first `SkinTrackRepository`.
 *
 * **Today:** use `createLocalSkinTrackRepository()` for all reads/writes. After sign-in, cloud backup
 * is `syncLocalBundleToSupabase` in `lib/data/sync.ts` (push-only, numeric local ids preserved inside
 * JSON `payload`). Do not wire this factory into `SkinTrackProvider` until pull/merge and id mapping exist.
 */
export function createSupabaseSkinTrackRepository(): SkinTrackRepository {
  return {
    loadRecords: async () => [],
    saveRecord: (_input: NewSkinTrackRecordInput): Promise<SaveRecordResult> => notConfigured(),
    replaceAllRecords: (_records: SkinTrackRecord[]): Promise<PersistResult> => notConfigured(),
    getProfile: () => emptyUserProfile(),
    setProfile: () => notConfigured(),
    getMedicationCatalog: (): MedicationCatalogItem[] => defaultMedicationCatalog(),
    setMedicationCatalog: (_items: MedicationCatalogItem[]) => notConfigured(),
    getMedDailyByDate: (): Record<string, DailyMedCheckoff> => ({}),
    setMedDailyByDate: (_map: Record<string, DailyMedCheckoff>) => notConfigured(),
    getLesions: (): Lesion[] => [],
    setLesions: () => notConfigured(),
    upsertLesion: () => notConfigured(),
    buildExport: (records: SkinTrackRecord[], profile: UserProfile): SkinTrackExportV1 =>
      buildExportPayload(records, profile),
    importBundle: async (raw: unknown, _mergeWithExisting: SkinTrackRecord[]): Promise<ImportBundleResult> => {
      const parsed = parseSkinTrackImport(raw)
      if (!parsed.ok) return { ok: false, error: parsed.error }
      return { ok: false, error: "Bundle import requires the local repository." }
    },
    getWebhookUrl: () => "",
    setWebhookUrl: () => notConfigured(),
    getApiKey: () => "",
    setApiKey: () => notConfigured(),
  }
}
