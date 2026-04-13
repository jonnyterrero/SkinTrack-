const ANGLE_BRACKET_RE = /[<>]/g

export function sanitizeText(input: string): string {
  return input.replace(ANGLE_BRACKET_RE, "").trim()
}

export function sanitizeObject<T extends Record<string, unknown>>(obj: T): T {
  const result = { ...obj }
  for (const key of Object.keys(result)) {
    const value = result[key]
    if (typeof value === "string") {
      ;(result as Record<string, unknown>)[key] = sanitizeText(value)
    } else if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      ;(result as Record<string, unknown>)[key] = sanitizeObject(
        value as Record<string, unknown>,
      )
    }
  }
  return result
}
