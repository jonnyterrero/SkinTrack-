import { z } from "zod"

export const profileUpdateSchema = z.object({
  display_name: z.string().max(100).optional(),
  skintrack_profile: z
    .object({
      name: z.string().max(100).default(""),
      age: z.string().max(10).default(""),
      gender: z.string().max(30).default(""),
      skinType: z.string().max(50).default(""),
      conditions: z.string().max(500).default(""),
      medications: z.string().max(500).default(""),
      allergies: z.string().max(500).default(""),
      notes: z.string().max(2000).default(""),
    })
    .passthrough()
    .optional(),
})

export type ProfileUpdateInput = z.infer<typeof profileUpdateSchema>
