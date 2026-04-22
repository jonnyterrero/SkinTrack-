import { z } from "zod"

export const createUserAllergySchema = z.object({
  allergen: z.string().min(1).max(200),
  severity: z.string().max(100).nullable().optional(),
  notes: z.string().max(1000).nullable().optional(),
})

export const updateUserAllergySchema = createUserAllergySchema.partial()

export type CreateUserAllergyInput = z.infer<typeof createUserAllergySchema>
export type UpdateUserAllergyInput = z.infer<typeof updateUserAllergySchema>
