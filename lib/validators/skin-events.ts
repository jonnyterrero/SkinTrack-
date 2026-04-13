import { z } from "zod"

export const createSkinEventSchema = z.object({
  lesion_id: z.string().uuid(),
  client_numeric_id: z.number().int().optional(),
  ts: z.string().datetime(),
  severity_0_4: z.number().int().min(0).max(4),
  location_id: z.string().min(1).max(100),
  itch: z.number().int().min(0).max(10),
  pain: z.number().int().min(0).max(10),
  burning: z.number().int().min(0).max(10),
  dryness: z.number().int().min(0).max(10),
  stress: z.number().int().min(0).max(10),
  sleep_hours: z.number().min(0).max(24),
  sleep_quality: z.number().int().min(1).max(5),
  metrics_schema_version: z.number().int().default(1),
  notes: z.string().max(2000).nullable().optional(),
})

export const updateSkinEventSchema = createSkinEventSchema.partial().omit({
  lesion_id: true,
})

export type CreateSkinEventInput = z.infer<typeof createSkinEventSchema>
export type UpdateSkinEventInput = z.infer<typeof updateSkinEventSchema>
