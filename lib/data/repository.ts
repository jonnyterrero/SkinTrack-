import type { DailyMedCheckoff, MedicationCatalogItem } from "@/lib/domain/medications"
import type { NewSkinTrackRecordInput, SkinTrackExportV1, SkinTrackRecord, UserProfile } from "@/lib/types"

export type SaveRecordResult =
  | { ok: true; record: SkinTrackRecord }
  | { ok: false; error: string; code?: "QUOTA" | "IDB" | "VALIDATION" }

export type PersistResult = { ok: true } | { ok: false; error: string; code?: "QUOTA" | "IDB" | "VALIDATION" }

export type ImportBundleResult =
  | { ok: true; records: SkinTrackRecord[]; profile: UserProfile }
  | { ok: false; error: string }

/**
 * Persistence abstraction. Production path: `createLocalSkinTrackRepository()`.
 *
 * Optional cloud backup uses `syncLocalBundleToSupabase` (push-only; see `lib/data/sync.ts`), not
 * `createSupabaseSkinTrackRepository()` — the latter remains a stub until a remote-first model ships.
 */
export interface SkinTrackRepository {
  loadRecords(): Promise<SkinTrackRecord[]>
  saveRecord(input: NewSkinTrackRecordInput): Promise<SaveRecordResult>
  replaceAllRecords(records: SkinTrackRecord[]): Promise<PersistResult>
  getProfile(): UserProfile
  setProfile(profile: UserProfile): void
  getMedicationCatalog(): MedicationCatalogItem[]
  setMedicationCatalog(items: MedicationCatalogItem[]): void
  getMedDailyByDate(): Record<string, DailyMedCheckoff>
  setMedDailyByDate(map: Record<string, DailyMedCheckoff>): void
  buildExport(records: SkinTrackRecord[], profile: UserProfile): SkinTrackExportV1
  importBundle(raw: unknown, mergeWithExisting: SkinTrackRecord[]): Promise<ImportBundleResult>
  getWebhookUrl(): string
  setWebhookUrl(url: string): void
  getApiKey(): string
  setApiKey(key: string): void
}
