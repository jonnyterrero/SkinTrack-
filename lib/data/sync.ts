import type { SupabaseClient } from "@supabase/supabase-js"
import { dataUrlToBlob } from "@/lib/data/blob-utils"
import type { SkinTrackExportV1 } from "@/lib/types"
import { isImageRecord, isSymptomRecord } from "@/lib/types"

export type SyncResult = { ok: true } | { ok: false; error: string }

/**
 * ## Cloud sync strategy (current)
 *
 * - **Direction:** Push-only. There is no pull/restore from Supabase into the local app yet.
 * - **Mechanism:** Full replace per user: delete all `records` for `auth.uid()`, then insert rows
 *   built from the local export bundle (`syncLocalBundleToSupabase`).
 * - **Local vs server IDs:** The app uses **numeric** `SkinTrackRecord.id` (e.g. `Date.now()`)
 *   in IndexedDB/localStorage. Each Supabase `public.records` row gets a **new UUID** primary key.
 *   The original client id remains inside `payload` (`payload.symptom` / `payload.image`), which
 *   embeds the full exported record including `id`. Future bidirectional sync can match on that
 *   field or add an explicit `client_record_id` column — not implemented here.
 *
 * For reads/writes during normal app use, always use `createLocalSkinTrackRepository()`; cloud is
 * optional backup via this function after magic-link sign-in.
 */
export async function syncLocalBundleToSupabase(
  supabase: SupabaseClient,
  bundle: SkinTrackExportV1,
): Promise<SyncResult> {
  const {
    data: { user },
    error: userErr,
  } = await supabase.auth.getUser()
  if (userErr || !user) {
    return { ok: false, error: userErr?.message ?? "Sign in required to sync." }
  }

  const skintrackProfile = {
    ...bundle.profile,
    medicationCatalog: bundle.medicationCatalog ?? [],
    medDailyByDate: bundle.medDailyByDate ?? {},
  }

  const { error: profileErr } = await supabase.from("profiles").upsert(
    {
      id: user.id,
      display_name: bundle.profile.name || null,
      skintrack_profile: skintrackProfile,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "id" },
  )
  if (profileErr) {
    return { ok: false, error: profileErr.message }
  }

  const { error: delErr } = await supabase.from("records").delete().eq("user_id", user.id)
  if (delErr) {
    return { ok: false, error: delErr.message }
  }

  for (const rec of bundle.records) {
    if (isSymptomRecord(rec)) {
      const { error } = await supabase.from("records").insert({
        user_id: user.id,
        ts: rec.timestamp,
        record_type: "symptom",
        payload: { symptom: rec, exportVersion: bundle.version },
        image_storage_path: null,
      })
      if (error) return { ok: false, error: error.message }
      continue
    }

    if (isImageRecord(rec)) {
      let storagePath: string | null = null
      if (rec.image?.startsWith("data:") && rec.filename) {
        const path = `${user.id}/${rec.id}/${rec.filename}`
        const blob = dataUrlToBlob(rec.image)
        const { error: upErr } = await supabase.storage.from("skintrack-images").upload(path, blob, {
          upsert: true,
          contentType: blob.type || "image/jpeg",
        })
        if (upErr) {
          return { ok: false, error: upErr.message }
        }
        storagePath = path
      }

      const { error } = await supabase.from("records").insert({
        user_id: user.id,
        ts: rec.timestamp,
        record_type: "image",
        payload: { image: { ...rec, image: undefined }, exportVersion: bundle.version },
        image_storage_path: storagePath,
      })
      if (error) return { ok: false, error: error.message }
    }
  }

  return { ok: true }
}
