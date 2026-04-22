import { NextResponse } from "next/server"
import {
  requireAuthAndRateLimit,
  dbError,
  apiError,
} from "@/lib/api/helpers"
import { getSupabaseAdminClient } from "@/lib/supabase/admin-client"

/**
 * DELETE /api/account — App Store / Play Store compliance.
 * Hard-deletes all user-owned data and the auth user row.
 */
export async function DELETE() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const bucket = supabase.storage.from("skintrack-images")
  const { data: files, error: listErr } = await bucket.list(user.id, {
    limit: 1000,
  })
  if (!listErr && files && files.length > 0) {
    const paths: string[] = []
    for (const file of files) {
      if (file.id === null) {
        const { data: sub } = await bucket.list(`${user.id}/${file.name}`, {
          limit: 1000,
        })
        if (sub) {
          for (const child of sub) {
            paths.push(`${user.id}/${file.name}/${child.name}`)
          }
        }
      } else {
        paths.push(`${user.id}/${file.name}`)
      }
    }
    if (paths.length > 0) {
      await bucket.remove(paths)
    }
  }

  const admin = getSupabaseAdminClient()
  const { error } = await admin.auth.admin.deleteUser(user.id)
  if (error) {
    return apiError("DB_ERROR", `Account deletion failed: ${error.message}`, 500)
  }

  return new NextResponse(null, { status: 204 })
}

/**
 * GET /api/account — returns a dump of user-owned data for export.
 */
export async function GET() {
  const { supabase, user, error: authError } = await requireAuthAndRateLimit()
  if (authError) return authError

  const [
    profile,
    prefs,
    conditions,
    allergies,
    lesions,
    skinEvents,
    medCatalog,
    lesionMedications,
    eventMedications,
    eventTriggers,
    eventProducts,
    eventImages,
    eventMetrics,
    lesionLocations,
    exportsRows,
  ] = await Promise.all([
    supabase.from("profiles").select("*").eq("id", user.id).maybeSingle(),
    supabase.from("app_preferences").select("*").eq("user_id", user.id).maybeSingle(),
    supabase.from("user_conditions").select("*").eq("user_id", user.id),
    supabase.from("user_allergies").select("*").eq("user_id", user.id),
    supabase.from("lesions").select("*").eq("user_id", user.id),
    supabase.from("skin_events").select("*").eq("user_id", user.id),
    supabase.from("med_catalog").select("*").eq("user_id", user.id),
    supabase.from("lesion_medications").select("*").eq("user_id", user.id),
    supabase.from("event_medications").select("*").eq("user_id", user.id),
    supabase.from("event_triggers").select("*").eq("user_id", user.id),
    supabase.from("event_products").select("*").eq("user_id", user.id),
    supabase.from("event_images").select("*").eq("user_id", user.id),
    supabase.from("event_metrics").select("*").eq("user_id", user.id),
    supabase.from("lesion_locations").select("*").eq("user_id", user.id),
    supabase.from("exports").select("*").eq("user_id", user.id),
  ])

  for (const r of [
    profile, prefs, conditions, allergies,
    lesions, skinEvents, medCatalog, lesionMedications,
    eventMedications, eventTriggers, eventProducts,
    eventImages, eventMetrics, lesionLocations, exportsRows,
  ]) {
    if (r.error) return dbError(r.error.message)
  }

  return NextResponse.json({
    exportVersion: "2026-04-20.v1",
    exportedAt: new Date().toISOString(),
    user: { id: user.id, email: user.email ?? null },
    profile: profile.data,
    appPreferences: prefs.data,
    userConditions: conditions.data ?? [],
    userAllergies: allergies.data ?? [],
    lesions: lesions.data ?? [],
    skinEvents: skinEvents.data ?? [],
    medCatalog: medCatalog.data ?? [],
    lesionMedications: lesionMedications.data ?? [],
    eventMedications: eventMedications.data ?? [],
    eventTriggers: eventTriggers.data ?? [],
    eventProducts: eventProducts.data ?? [],
    eventImages: eventImages.data ?? [],
    eventMetrics: eventMetrics.data ?? [],
    lesionLocations: lesionLocations.data ?? [],
    exports: exportsRows.data ?? [],
  })
}
