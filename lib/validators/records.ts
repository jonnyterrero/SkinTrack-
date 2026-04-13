import { z } from "zod"

export const createRecordSchema = z.object({
  ts: z.string().datetime().optional(),
  record_type: z.enum(["symptom", "image"]),
  payload: z.record(z.unknown()).default({}),
  image_storage_path: z.string().max(500).nullable().optional(),
})

export const updateRecordSchema = z.object({
  ts: z.string().datetime().optional(),
  payload: z.record(z.unknown()).optional(),
  image_storage_path: z.string().max(500).nullable().optional(),
})

export type CreateRecordInput = z.infer<typeof createRecordSchema>
export type UpdateRecordInput = z.infer<typeof updateRecordSchema>
