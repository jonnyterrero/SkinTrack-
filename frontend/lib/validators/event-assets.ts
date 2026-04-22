import { z } from "zod"

// image_kind_enum: raw, processed, mask, overlay (NO thumbnail).
export const createEventImageSchema = z.object({
  skin_event_id: z.string().uuid(),
  storage_path: z.string().min(1).max(500),
  kind: z.enum(["raw", "processed", "mask", "overlay"]).default("raw"),
  width: z.number().int().positive().nullable().optional(),
  height: z.number().int().positive().nullable().optional(),
  mime_type: z.string().max(100).nullable().optional(),
  captured_at: z.string().datetime().nullable().optional(),
  processing_status: z
    .enum(["pending", "processing", "succeeded", "failed"])
    .default("pending"),
  failure_reason: z.string().max(500).nullable().optional(),
})

export const updateEventImageSchema = createEventImageSchema
  .partial()
  .omit({ skin_event_id: true })

export const upsertEventMetricsSchema = z.object({
  skin_event_id: z.string().uuid(),
  metrics_schema_version: z.number().int().default(1),
  area_cm2: z.number().nullable().optional(),
  redness_index: z.number().nullable().optional(),
  border_irregularity: z.number().nullable().optional(),
  asymmetry: z.number().nullable().optional(),
  delta_e: z.number().nullable().optional(),
  raw_area_px: z.number().nullable().optional(),
  raw_perimeter_px: z.number().nullable().optional(),
  cm_per_px: z.number().nullable().optional(),
  scale_mode: z.enum(["aruco", "fallback", "manual"]).default("fallback"),
  segmentation_mode: z.enum(["kmeans", "grabcut", "unet", "none"]).default("none"),
  calibration_applied: z.boolean().default(false),
  confidence_score: z.number().min(0).max(1).nullable().optional(),
})

export const createExportSchema = z.object({
  lesion_id: z.string().uuid().nullable().optional(),
  export_type: z.enum(["pdf_summary", "csv", "json"]),
  storage_path: z.string().min(1).max(500),
  start_ts: z.string().datetime().nullable().optional(),
  end_ts: z.string().datetime().nullable().optional(),
})

export type CreateEventImageInput = z.infer<typeof createEventImageSchema>
export type UpdateEventImageInput = z.infer<typeof updateEventImageSchema>
export type UpsertEventMetricsInput = z.infer<typeof upsertEventMetricsSchema>
export type CreateExportInput = z.infer<typeof createExportSchema>
