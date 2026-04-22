// Backend (Supabase) row shapes — aligned to the live iOS schema.
// Source of truth: claude-supabase/supabase/migrations/20260420042221..20260420042354*.sql

export type Condition = {
  id: string
  slug: string
  name: string
  display_name: string
  supports_imaging: boolean
  supports_body_map: boolean
  created_at: string
}

export type TriggerTaxonomyRow = {
  key: string
  label: string
  value_type: "boolean" | "text" | "number" | "enum"
  created_at: string
}

export type BodyView = "front" | "back"
export type BodySide = "left" | "right" | "midline" | "unknown"

// The DB uses plain text for body_region (no check constraint).
// The client maintains the canonical list so UI stays consistent.
export type BodyRegion =
  | "head" | "face" | "neck" | "chest" | "abdomen"
  | "back_upper" | "back_lower" | "shoulder"
  | "arm_upper" | "arm_lower" | "elbow" | "hand" | "wrist"
  | "hip" | "glute" | "thigh" | "knee" | "leg_lower" | "ankle" | "foot"
  | "genital" | "scalp" | "other"

export type LesionLocation = {
  id: string
  user_id: string
  lesion_id: string
  body_view: BodyView
  body_region: string
  side: BodySide
  loc_x: number
  loc_y: number
  created_at: string
}

// Shared med_category_enum (used by med_catalog.category AND event_products.product_type).
export type MedCategory =
  | "topical" | "oral" | "injection" | "otc"
  | "moisturizer" | "cleanser" | "diet"
  | "environmental" | "avoidance" | "home_remedy" | "other"

export type MedCatalogItem = {
  id: string
  user_id: string
  name: string
  category: MedCategory
  dose: string | null
  frequency: string | null
  morning: boolean
  afternoon: boolean
  evening: boolean
  is_prescription: boolean
  prescribed_by: string | null
  start_date: string | null
  end_date: string | null
  notes: string | null
  active: boolean
  created_at: string
  updated_at: string
}

export type LesionMedicationScope = "global" | "lesion_specific"

export type LesionMedication = {
  id: string
  user_id: string
  lesion_id: string
  med_catalog_id: string
  scope: LesionMedicationScope
}

export type EventMedication = {
  id: string
  user_id: string
  skin_event_id: string
  med_catalog_id: string
  taken: boolean
  amount_text: string | null
  missed_reason: string | null
  notes: string | null
}

export type EventTrigger = {
  id: string
  user_id: string
  skin_event_id: string
  trigger_key: string
  trigger_value_text: string | null
}

export type EventProduct = {
  id: string
  user_id: string
  skin_event_id: string
  product_name: string
  product_type: MedCategory
  first_use: boolean
  used: boolean
  perceived_benefit: number | null
  adverse_reaction: boolean
  notes: string | null
}

export type ImageKind = "raw" | "processed" | "mask" | "overlay"
export type ProcessingStatus = "pending" | "processing" | "succeeded" | "failed"

export type EventImage = {
  id: string
  user_id: string
  skin_event_id: string
  storage_path: string
  kind: ImageKind
  width: number | null
  height: number | null
  mime_type: string | null
  captured_at: string | null
  processing_status: ProcessingStatus
  failure_reason: string | null
}

export type ScaleMode = "aruco" | "fallback" | "manual"
export type SegmentationMode = "kmeans" | "grabcut" | "unet" | "none"

export type EventMetrics = {
  id: string
  user_id: string
  skin_event_id: string
  metrics_schema_version: number
  area_cm2: number | null
  redness_index: number | null
  border_irregularity: number | null
  asymmetry: number | null
  delta_e: number | null
  raw_area_px: number | null
  raw_perimeter_px: number | null
  cm_per_px: number | null
  scale_mode: ScaleMode
  segmentation_mode: SegmentationMode
  calibration_applied: boolean
  confidence_score: number | null
}

export type ExportType = "pdf_summary" | "csv" | "json"

export type ExportRow = {
  id: string
  user_id: string
  lesion_id: string | null
  export_type: ExportType
  storage_path: string
  start_ts: string | null
  end_ts: string | null
  created_at: string
}

export type AppPreferences = {
  user_id: string
  completed_onboarding: boolean
  preferred_log_time: string | null
  reminders_enabled: boolean
  quiet_hours_start: string | null
  quiet_hours_end: string | null
  units: "metric" | "imperial"
  theme: "system" | "light" | "dark"
  consent_version: string
  privacy_policy_url: string | null
  created_at: string
  updated_at: string
}

export type Profile = {
  id: string
  display_name: string | null
  email: string | null
  onboarding_completed_at: string | null
  consent_acknowledged_at: string | null
  consent_version: string | null
  symptom_scale_version: string | null
  clinic_notes: string | null
  skintrack_profile: Record<string, unknown>
  created_at: string
  updated_at: string
}

export type UserCondition = {
  id: string
  user_id: string
  condition_id: string
  source: "self_reported" | "clinician_diagnosed"
  created_at: string
}

export type UserAllergy = {
  id: string
  user_id: string
  allergen: string
  severity: string | null
  notes: string | null
  created_at: string
}

export const CURRENT_CONSENT_VERSION = "2026-04-20.v1"
export const CURRENT_SYMPTOM_SCALE_VERSION = "2026-04-20.v1"
