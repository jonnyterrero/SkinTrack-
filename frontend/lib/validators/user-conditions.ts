import { z } from "zod"

export const CONDITION_SOURCES = ["self_reported", "clinician_diagnosed"] as const

export const createUserConditionSchema = z.object({
  condition_id: z.string().uuid(),
  source: z.enum(CONDITION_SOURCES).default("self_reported"),
})

export type CreateUserConditionInput = z.infer<typeof createUserConditionSchema>
