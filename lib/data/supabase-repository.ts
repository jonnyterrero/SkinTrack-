import type { SkinTrackRepository } from "@/lib/data/repository"
import {
  emptyUserProfile,
  EXPORT_SCHEMA_VERSION,
  type NewSkinTrackRecordInput,
  type SkinTrackExportV1,
  type SkinTrackRecord,
  type UserProfile,
} from "@/lib/types"

const notConfigured = (): never => {
  throw new Error(
    "SupabaseRepository is not wired yet. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY, then implement sync against supabase/schema.sql.",
  )
}

/**
 * Placeholder for a future `SkinTrackRepository` backed by Supabase Auth + Postgres + Storage.
 * Keeps the same interface as the local repository for a later swap or hybrid "local-first + sync" mode.
 */
export function createSupabaseSkinTrackRepository(): SkinTrackRepository {
  return {
    loadRecords: () => notConfigured(),
    saveRecord: (_input: NewSkinTrackRecordInput) => notConfigured(),
    replaceAllRecords: (_records: SkinTrackRecord[]) => notConfigured(),
    getProfile: () => emptyUserProfile(),
    setProfile: () => notConfigured(),
    buildExport: (_records: SkinTrackRecord[], profile: UserProfile): SkinTrackExportV1 => ({
      version: EXPORT_SCHEMA_VERSION,
      exportDate: new Date().toISOString(),
      records: _records,
      profile,
    }),
    importBundle: () => notConfigured(),
    getWebhookUrl: () => "",
    setWebhookUrl: () => notConfigured(),
    getApiKey: () => "",
    setApiKey: () => notConfigured(),
  }
}
