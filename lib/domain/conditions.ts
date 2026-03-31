/** Single source of truth for condition pickers (symptom form, exports, setup). */

export const SKIN_CONDITIONS = [
  "eczema",
  "psoriasis",
  "guttate psoriasis",
  "keratosis pilaris",
  "cystic/hormonal acne",
  "melanoma",
  "vitiligo",
  "contact dermatitis",
  "cold sores",
] as const

export type SkinConditionId = (typeof SKIN_CONDITIONS)[number]
