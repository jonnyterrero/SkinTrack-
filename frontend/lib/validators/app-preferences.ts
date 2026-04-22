import { z } from "zod"

const timeOrNull = z
  .string()
  .regex(/^\d{2}:\d{2}(:\d{2})?$/)
  .nullable()
  .optional()

export const updateAppPreferencesSchema = z.object({
  completed_onboarding: z.boolean().optional(),
  consent_version: z.string().max(64).optional(),
  privacy_policy_url: z.string().url().max(512).nullable().optional(),
  preferred_log_time: timeOrNull,
  reminders_enabled: z.boolean().optional(),
  quiet_hours_start: timeOrNull,
  quiet_hours_end: timeOrNull,
  units: z.enum(["metric", "imperial"]).optional(),
  theme: z.enum(["system", "light", "dark"]).optional(),
})

export type UpdateAppPreferencesInput = z.infer<typeof updateAppPreferencesSchema>
