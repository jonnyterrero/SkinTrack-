import { z } from "zod"

// Mirrors med_category_enum. Used by med_catalog.category and event_products.product_type.
export const MED_CATEGORIES = [
  "topical", "oral", "injection", "otc",
  "moisturizer", "cleanser",
  "diet", "environmental", "avoidance", "home_remedy", "other",
] as const

export const createMedicationSchema = z.object({
  name: z.string().min(1).max(200),
  category: z.enum(MED_CATEGORIES),
  dose: z.string().max(200).nullable().optional(),
  frequency: z.string().max(200).nullable().optional(),
  morning: z.boolean().default(false),
  afternoon: z.boolean().default(false),
  evening: z.boolean().default(false),
  is_prescription: z.boolean().default(false),
  prescribed_by: z.string().max(200).nullable().optional(),
  start_date: z.string().date().nullable().optional(),
  end_date: z.string().date().nullable().optional(),
  notes: z.string().max(2000).nullable().optional(),
  active: z.boolean().default(true),
})

export const updateMedicationSchema = createMedicationSchema.partial()

export const createEventMedicationSchema = z.object({
  skin_event_id: z.string().uuid(),
  med_catalog_id: z.string().uuid(),
  taken: z.boolean(),
  amount_text: z.string().max(200).nullable().optional(),
  missed_reason: z.string().max(500).nullable().optional(),
  notes: z.string().max(2000).nullable().optional(),
})

export const updateEventMedicationSchema = createEventMedicationSchema
  .partial()
  .omit({ skin_event_id: true, med_catalog_id: true })

export const createLesionMedicationSchema = z.object({
  lesion_id: z.string().uuid(),
  med_catalog_id: z.string().uuid(),
  scope: z.enum(["global", "lesion_specific"]),
})

export type CreateMedicationInput = z.infer<typeof createMedicationSchema>
export type UpdateMedicationInput = z.infer<typeof updateMedicationSchema>
export type CreateEventMedicationInput = z.infer<typeof createEventMedicationSchema>
export type UpdateEventMedicationInput = z.infer<typeof updateEventMedicationSchema>
export type CreateLesionMedicationInput = z.infer<typeof createLesionMedicationSchema>
