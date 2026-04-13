import { z } from "zod"

const ALLOWED_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/heic",
  "image/heif",
] as const

export const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10 MB
export const MAX_DIMENSION = 8192

export const uploadMetadataSchema = z.object({
  record_id: z.string().uuid().optional(),
  filename: z.string().min(1).max(255),
})

export function validateMimeType(contentType: string | null): contentType is string {
  if (!contentType) return false
  return (ALLOWED_MIME_TYPES as readonly string[]).includes(contentType)
}

const MAGIC_BYTES: Record<string, number[]> = {
  "image/jpeg": [0xff, 0xd8, 0xff],
  "image/png": [0x89, 0x50, 0x4e, 0x47],
  "image/webp": [0x52, 0x49, 0x46, 0x46],
}

export function validateMagicBytes(
  buffer: ArrayBuffer,
  claimedType: string,
): boolean {
  const expected = MAGIC_BYTES[claimedType]
  if (!expected) return true // HEIC/HEIF — no simple magic-byte check
  const bytes = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 12))
  return expected.every((b, i) => bytes[i] === b)
}
