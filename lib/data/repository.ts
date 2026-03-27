import type { NewSkinTrackRecordInput, SkinTrackExportV1, SkinTrackRecord, UserProfile } from "@/lib/types"

export type SaveRecordResult =
  | { ok: true; record: SkinTrackRecord }
  | { ok: false; error: string; code?: "QUOTA" | "IDB" | "VALIDATION" }

export type PersistResult = { ok: true } | { ok: false; error: string; code?: "QUOTA" | "IDB" | "VALIDATION" }

export type ImportBundleResult =
  | { ok: true; records: SkinTrackRecord[]; profile: UserProfile }
  | { ok: false; error: string }

/**
 * Persistence abstraction. Local implementation today; Supabase implementation later.
 */
export interface SkinTrackRepository {
  loadRecords(): Promise<SkinTrackRecord[]>
  saveRecord(input: NewSkinTrackRecordInput): Promise<SaveRecordResult>
  /** Replace all records (e.g. after merge import). Persists and returns stored shape. */
  replaceAllRecords(records: SkinTrackRecord[]): Promise<PersistResult>
  getProfile(): UserProfile
  setProfile(profile: UserProfile): void
  buildExport(records: SkinTrackRecord[], profile: UserProfile): SkinTrackExportV1
  importBundle(raw: unknown, mergeWithExisting: SkinTrackRecord[]): Promise<ImportBundleResult>
  getWebhookUrl(): string
  setWebhookUrl(url: string): void
  getApiKey(): string
  setApiKey(key: string): void
}
