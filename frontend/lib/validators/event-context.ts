import { z } from "zod"
import { MED_CATEGORIES } from "./medications"

// event_triggers.trigger_key is plain text, cross-referenced against trigger_taxonomy.
// Client must fetch the canonical list from /api/triggers.
export const createEventTriggerSchema = z.object({
  skin_event_id: z.string().uuid(),
  trigger_key: z.string().min(1).max(64),
  trigger_value_text: z.string().max(500).nullable().optional(),
})

export const updateEventTriggerSchema = createEventTriggerSchema
  .partial()
  .omit({ skin_event_id: true })

// event_products.product_type uses shared med_category_enum.
export const createEventProductSchema = z.object({
  skin_event_id: z.string().uuid(),
  product_name: z.string().min(1).max(200),
  product_type: z.enum(MED_CATEGORIES),
  first_use: z.boolean().default(false),
  used: z.boolean().default(true),
  perceived_benefit: z.number().int().min(-2).max(2).nullable().optional(),
  adverse_reaction: z.boolean().default(false),
  notes: z.string().max(1000).nullable().optional(),
})

export const updateEventProductSchema = createEventProductSchema
  .partial()
  .omit({ skin_event_id: true })

export type CreateEventTriggerInput = z.infer<typeof createEventTriggerSchema>
export type UpdateEventTriggerInput = z.infer<typeof updateEventTriggerSchema>
export type CreateEventProductInput = z.infer<typeof createEventProductSchema>
export type UpdateEventProductInput = z.infer<typeof updateEventProductSchema>
