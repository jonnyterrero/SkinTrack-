import { STORAGE_KEYS } from "@/lib/data/keys"
import { buildExportPayload, parseSkinTrackImport } from "@/lib/data/export-import"
import type { DailyMedCheckoff, MedicationCatalogItem } from "@/lib/domain/medications"
import { defaultMedicationCatalog } from "@/lib/domain/medications"
import { dataUrlToBlob, generateImageRef } from "@/lib/data/blob-utils"
import { getImageBlob, putImageBlob } from "@/lib/data/idb"
import { hydrateRecordsForUi, migrateLegacyRecords } from "@/lib/data/migrate"
import type { SkinTrackRepository, SaveRecordResult, ImportBundleResult, PersistResult } from "@/lib/data/repository"
import { METRICS_SCHEMA_VERSION } from "@/lib/domain/skin-event-metrics"
import {
  emptyUserProfile,
  EXPORT_SCHEMA_VERSION,
  type Lesion,
  type NewSkinTrackRecordInput,
  type PersistedImageRow,
  type PersistedRow,
  type SkinEventRecord,
  type SkinTrackExportV1,
  type SkinTrackRecord,
  type SymptomTrackRecord,
  type UserProfile,
} from "@/lib/types"

function parseRecordsJson(raw: string | null): unknown[] {
  if (!raw) return []
  try {
    const data = JSON.parse(raw) as unknown
    return Array.isArray(data) ? data : []
  } catch {
    return []
  }
}

function persistRowsToLocalStorage(rows: PersistedRow[]): PersistResult {
  try {
    localStorage.setItem(STORAGE_KEYS.records, JSON.stringify(rows))
    return { ok: true }
  } catch (e) {
    const name = e instanceof DOMException ? e.name : ""
    if (name === "QuotaExceededError" || (e as Error)?.message?.includes("quota")) {
      return {
        ok: false,
        error:
          "Browser storage is full. Export a backup, then remove old images or clear site data for this app.",
        code: "QUOTA",
      }
    }
    return { ok: false, error: (e as Error)?.message ?? "Failed to save records." }
  }
}

async function loadPersistedRows(): Promise<PersistedRow[]> {
  const raw = parseRecordsJson(localStorage.getItem(STORAGE_KEYS.records))
  const { rows, mutated } = await migrateLegacyRecords(raw)
  if (mutated) {
    persistRowsToLocalStorage(rows as PersistedRow[])
  }
  return rows as PersistedRow[]
}

async function ensureImageBlobInIdb(record: SkinTrackRecord): Promise<string> {
  if (record.type !== "image") {
    throw new Error("Expected image record")
  }
  const ref = record.imageRef ?? generateImageRef()
  const existing = await getImageBlob(ref)
  if (existing) return ref
  if (record.image?.startsWith("data:")) {
    await putImageBlob(ref, dataUrlToBlob(record.image))
  }
  return ref
}

function toPersistedRow(record: SkinTrackRecord, imageRef: string | undefined): PersistedRow {
  if (record.type === "symptom" || record.type === "skin_event") {
    return record
  }
  return {
    id: record.id,
    timestamp: record.timestamp,
    type: "image",
    filename: record.filename,
    imageRef: imageRef ?? record.imageRef ?? generateImageRef(),
    ...(record.metadata ? { metadata: record.metadata } : {}),
  }
}

async function recordsToPersistedRows(records: SkinTrackRecord[]): Promise<PersistedRow[]> {
  const out: PersistedRow[] = []
  for (const r of records) {
    if (r.type === "symptom" || r.type === "skin_event") {
      out.push(r)
      continue
    }
    const ref = await ensureImageBlobInIdb({ ...r, imageRef: r.imageRef })
    out.push(toPersistedRow({ ...r, imageRef: ref }, ref))
  }
  return out
}

export function createLocalSkinTrackRepository(): SkinTrackRepository {
  const repo: SkinTrackRepository = {
    async loadRecords(): Promise<SkinTrackRecord[]> {
      if (typeof window === "undefined") return []
      const persisted = await loadPersistedRows()
      return hydrateRecordsForUi(persisted)
    },

    async saveRecord(input: NewSkinTrackRecordInput): Promise<SaveRecordResult> {
      const id = Date.now()
      const timestamp = new Date().toISOString()

      const persisted = await loadPersistedRows()

      if (input.type === "symptom") {
        const record: SymptomTrackRecord = { ...input, id, timestamp }
        const next: PersistedRow[] = [record, ...persisted]
        const pr = persistRowsToLocalStorage(next)
        if (!pr.ok) return pr
        return { ok: true, record }
      }

      if (input.type === "skin_event") {
        const record: SkinEventRecord = {
          ...input,
          id,
          timestamp,
          metricsSchemaVersion: METRICS_SCHEMA_VERSION,
        }
        const next: PersistedRow[] = [record, ...persisted]
        const pr = persistRowsToLocalStorage(next)
        if (!pr.ok) return pr
        return { ok: true, record }
      }

      const ref = generateImageRef()
      try {
        await putImageBlob(ref, dataUrlToBlob(input.image))
      } catch (e) {
        return {
          ok: false,
          error: (e as Error)?.message ?? "Could not store image locally.",
          code: "IDB",
        }
      }

      const record: SkinTrackRecord = {
        id,
        timestamp,
        type: "image",
        filename: input.filename,
        imageRef: ref,
        image: input.image,
        ...(input.metadata ? { metadata: input.metadata } : {}),
      }

      const imageRow: PersistedImageRow = {
        id,
        timestamp,
        type: "image",
        filename: input.filename,
        imageRef: ref,
        ...(input.metadata ? { metadata: input.metadata } : {}),
      }
      const pr = persistRowsToLocalStorage([imageRow, ...persisted])
      if (!pr.ok) return pr
      return { ok: true, record }
    },

    async replaceAllRecords(records: SkinTrackRecord[]): Promise<PersistResult> {
      try {
        const next = await recordsToPersistedRows(records)
        return persistRowsToLocalStorage(next)
      } catch (e) {
        return {
          ok: false,
          error: (e as Error)?.message ?? "Failed to persist records.",
          code: "IDB",
        }
      }
    },

    getProfile(): UserProfile {
      if (typeof window === "undefined") return emptyUserProfile()
      const raw = localStorage.getItem(STORAGE_KEYS.profile)
      if (!raw) return emptyUserProfile()
      try {
        return { ...emptyUserProfile(), ...JSON.parse(raw) } as UserProfile
      } catch {
        return emptyUserProfile()
      }
    },

    setProfile(profile: UserProfile): void {
      if (typeof window === "undefined") return
      try {
        localStorage.setItem(STORAGE_KEYS.profile, JSON.stringify(profile))
      } catch {
        /* ignore */
      }
    },

    getMedicationCatalog(): MedicationCatalogItem[] {
      if (typeof window === "undefined") return defaultMedicationCatalog()
      const raw = localStorage.getItem(STORAGE_KEYS.medicationCatalog)
      if (!raw) return defaultMedicationCatalog()
      try {
        const parsed = JSON.parse(raw) as unknown
        return Array.isArray(parsed) ? (parsed as MedicationCatalogItem[]) : defaultMedicationCatalog()
      } catch {
        return defaultMedicationCatalog()
      }
    },

    setMedicationCatalog(items: MedicationCatalogItem[]): void {
      if (typeof window === "undefined") return
      try {
        localStorage.setItem(STORAGE_KEYS.medicationCatalog, JSON.stringify(items))
      } catch {
        /* ignore */
      }
    },

    getMedDailyByDate(): Record<string, DailyMedCheckoff> {
      if (typeof window === "undefined") return {}
      const raw = localStorage.getItem(STORAGE_KEYS.medDailyByDate)
      if (!raw) return {}
      try {
        const parsed = JSON.parse(raw) as unknown
        return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
          ? (parsed as Record<string, DailyMedCheckoff>)
          : {}
      } catch {
        return {}
      }
    },

    setMedDailyByDate(map: Record<string, DailyMedCheckoff>): void {
      if (typeof window === "undefined") return
      try {
        localStorage.setItem(STORAGE_KEYS.medDailyByDate, JSON.stringify(map))
      } catch {
        /* ignore */
      }
    },

    getLesions(): Lesion[] {
      if (typeof window === "undefined") return []
      const raw = localStorage.getItem(STORAGE_KEYS.lesions)
      if (!raw) return []
      try {
        const parsed = JSON.parse(raw) as unknown
        return Array.isArray(parsed) ? (parsed as Lesion[]) : []
      } catch {
        return []
      }
    },

    setLesions(lesions: Lesion[]): void {
      if (typeof window === "undefined") return
      try {
        localStorage.setItem(STORAGE_KEYS.lesions, JSON.stringify(lesions))
      } catch {
        /* ignore */
      }
    },

    upsertLesion(lesion: Lesion): void {
      const list = repo.getLesions()
      const idx = list.findIndex((l) => l.id === lesion.id)
      const next = idx >= 0 ? list.map((l) => (l.id === lesion.id ? lesion : l)) : [lesion, ...list]
      repo.setLesions(next)
    },

    buildExport(records: SkinTrackRecord[], profile: UserProfile): SkinTrackExportV1 {
      return buildExportPayload(records, profile, {
        medicationCatalog: repo.getMedicationCatalog(),
        medDailyByDate: repo.getMedDailyByDate(),
        lesions: repo.getLesions(),
      })
    },

    async importBundle(raw: unknown, mergeWithExisting: SkinTrackRecord[]): Promise<ImportBundleResult> {
      const parsed = parseSkinTrackImport(raw)
      if (!parsed.ok) return { ok: false, error: parsed.error }

      const incoming = parsed.bundle.records.map((r) => ({ ...r }))

      for (const r of incoming) {
        if (r.type === "image" && r.image?.startsWith("data:")) {
          const ref = r.imageRef ?? generateImageRef()
          await putImageBlob(ref, dataUrlToBlob(r.image))
          r.imageRef = ref
        }
      }

      const merged: SkinTrackRecord[] = [...mergeWithExisting, ...incoming]
      const replace = await repo.replaceAllRecords(merged)
      if (!replace.ok) {
        return { ok: false, error: "Could not save imported data. Storage may be full." }
      }
      repo.setProfile(parsed.bundle.profile)
      if (parsed.bundle.medicationCatalog) {
        repo.setMedicationCatalog(parsed.bundle.medicationCatalog)
      }
      if (parsed.bundle.medDailyByDate) {
        repo.setMedDailyByDate(parsed.bundle.medDailyByDate)
      }
      if (parsed.bundle.lesions?.length) {
        const existing = repo.getLesions()
        const byId = new Map(existing.map((l) => [l.id, l]))
        for (const l of parsed.bundle.lesions) {
          byId.set(l.id, l)
        }
        repo.setLesions([...byId.values()])
      }
      const hydrated = await repo.loadRecords()
      return { ok: true, records: hydrated, profile: parsed.bundle.profile }
    },

    getWebhookUrl(): string {
      if (typeof window === "undefined") return ""
      return localStorage.getItem(STORAGE_KEYS.webhookUrl) ?? ""
    },

    setWebhookUrl(url: string): void {
      if (typeof window === "undefined") return
      localStorage.setItem(STORAGE_KEYS.webhookUrl, url)
    },

    getApiKey(): string {
      if (typeof window === "undefined") return ""
      return localStorage.getItem(STORAGE_KEYS.apiKey) ?? ""
    },

    setApiKey(key: string): void {
      if (typeof window === "undefined") return
      localStorage.setItem(STORAGE_KEYS.apiKey, key)
    },
  }

  return repo
}

export { EXPORT_SCHEMA_VERSION }
