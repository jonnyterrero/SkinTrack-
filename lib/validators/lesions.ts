import { z } from "zod"

export const createLesionSchema = z.object({
  id: z.string().uuid().optional(),
  label: z.string().min(1).max(200),
})

export const updateLesionSchema = z.object({
  label: z.string().min(1).max(200).optional(),
  archived_at: z.string().datetime().nullable().optional(),
})

export type CreateLesionInput = z.infer<typeof createLesionSchema>
export type UpdateLesionInput = z.infer<typeof updateLesionSchema>
