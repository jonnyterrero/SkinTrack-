import type { SupabaseClient } from "@supabase/supabase-js"
import { dataUrlToBlob } from "@/lib/data/blob-utils"
import type { SkinTrackExportV1 } from "@/lib/types"
import { isImageRecord, isSkinEventRecord, isSymptomRecord } from "@/lib/types"

export type SyncResult = { ok: true } | { ok: false; error: string }

/**
 * ## Cloud sync strategy (current)
 *
 * - **Direction:** Push-only. There is no pull/restore from Supabase into the local app yet.
 * - **Mechanism:** Full replace per user: delete `skin_events`, `lesions`, then `records` for
 *   `auth.uid()`, then re-insert from the export bundle. Lesions and skin events map to
 *   `public.lesions` / `public.skin_events`; symptom and image rows stay in `public.records`.
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

  const { error: delSkinErr } = await supabase.from("skin_events").delete().eq("user_id", user.id)
  if (delSkinErr) {
    return { ok: false, error: delSkinErr.message }
  }

  const { error: delLesionsErr } = await supabase.from("lesions").delete().eq("user_id", user.id)
  if (delLesionsErr) {
    return { ok: false, error: delLesionsErr.message }
  }

  const { error: delErr } = await supabase.from("records").delete().eq("user_id", user.id)
  if (delErr) {
    return { ok: false, error: delErr.message }
  }

  const bundleLesions = bundle.lesions ?? []
  const lesionById = new Map(bundleLesions.map((l) => [l.id, l]))
  for (const rec of bundle.records) {
    if (isSkinEventRecord(rec) && !lesionById.has(rec.lesionId)) {
      lesionById.set(rec.lesionId, {
        id: rec.lesionId,
        label: "Unlisted lesion",
        createdAt: rec.timestamp,
      })
    }
  }

  for (const l of lesionById.values()) {
    const { error: leErr } = await supabase.from("lesions").insert({
      id: l.id,
      user_id: user.id,
      label: l.label,
      created_at: l.createdAt,
      archived_at: l.archived ? new Date().toISOString() : null,
    })
    if (leErr) return { ok: false, error: leErr.message }
  }

  for (const rec of bundle.records) {
    if (isSkinEventRecord(rec)) {
      const { error } = await supabase.from("skin_events").insert({
        user_id: user.id,
        lesion_id: rec.lesionId,
        client_numeric_id: rec.id,
        ts: rec.timestamp,
        severity_0_4: rec.severity04,
        location_id: rec.locationId,
        itch: rec.itch,
        pain: rec.pain,
        burning: rec.burning,
        dryness: rec.dryness,
        stress: rec.stress,
        sleep_hours: rec.sleepHours,
        sleep_quality: rec.sleepQuality,
        metrics_schema_version: rec.metricsSchemaVersion,
        notes: rec.notes ?? null,
      })
      if (error) return { ok: false, error: error.message }
      continue
    }

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
