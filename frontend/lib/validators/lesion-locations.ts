import { z } from "zod"

export const BODY_VIEWS = ["front", "back"] as const
export const BODY_REGIONS = [
  "head", "face", "neck", "chest", "abdomen",
  "back_upper", "back_lower", "shoulder",
  "arm_upper", "arm_lower", "elbow", "hand", "wrist",
  "hip", "glute", "thigh", "knee", "leg_lower", "ankle", "foot",
  "genital", "scalp", "other",
] as const
// DB enum side_enum: 'left','right','midline','unknown' (no 'center').
export const BODY_SIDES = ["left", "right", "midline", "unknown"] as const

export const createLesionLocationSchema = z.object({
  lesion_id: z.string().uuid(),
  body_view: z.enum(BODY_VIEWS),
  body_region: z.enum(BODY_REGIONS),
  side: z.enum(BODY_SIDES).default("unknown"),
  loc_x: z.number().min(0).max(1),
  loc_y: z.number().min(0).max(1),
})

export const updateLesionLocationSchema = createLesionLocationSchema
  .partial()
  .omit({ lesion_id: true })

export type CreateLesionLocationInput = z.infer<typeof createLesionLocationSchema>
export type UpdateLesionLocationInput = z.infer<typeof updateLesionLocationSchema>
