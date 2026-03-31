import type { PersistedRow, SkinTrackRecord, SymptomTrackRecord } from "@/lib/types"
import { blobToDataUrl, dataUrlToBlob, generateImageRef } from "@/lib/data/blob-utils"
import { getImageBlob, putImageBlob } from "@/lib/data/idb"

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v)
}

/**
 * Normalizes parsed localStorage rows: moves inline image data URLs into IndexedDB.
 * Returns rows safe to persist in localStorage (no large image field).
 */
export async function migrateLegacyRecords(rows: unknown[]): Promise<{ rows: PersistedRow[]; mutated: boolean }> {
  let mutated = false
  const out: PersistedRow[] = []

  for (const row of rows) {
    if (!isPlainObject(row) || typeof row.type !== "string") continue

    if (row.type === "symptom") {
      out.push(row as unknown as SymptomTrackRecord)
      continue
    }

    if (row.type !== "image") continue

    const img = row as Record<string, unknown>
    const id = typeof img.id === "number" ? img.id : Number(img.id) || Date.now()
    const timestamp = typeof img.timestamp === "string" ? img.timestamp : new Date().toISOString()
    const filename = typeof img.filename === "string" ? img.filename : `image_${id}.jpg`

    const inline = typeof img.image === "string" ? img.image : null
    const existingRef = typeof img.imageRef === "string" ? img.imageRef : null

    if (inline && inline.startsWith("data:")) {
      const ref = existingRef ?? generateImageRef()
      const blob = dataUrlToBlob(inline)
      await putImageBlob(ref, blob)
      out.push({ id, timestamp, type: "image", filename, imageRef: ref })
      mutated = true
      continue
    }

    if (existingRef) {
      out.push({ id, timestamp, type: "image", filename, imageRef: existingRef })
      continue
    }

    // Broken image row — keep filename only to avoid data loss of metadata
    const ref = generateImageRef()
    out.push({ id, timestamp, type: "image", filename, imageRef: ref })
    mutated = true
  }

  return { rows: out, mutated }
}

/** Hydrate persisted rows with `image` data URLs for UI components. */
export async function hydrateRecordsForUi(rows: PersistedRow[]): Promise<SkinTrackRecord[]> {
  const result: SkinTrackRecord[] = []
  for (const row of rows) {
    if (row.type === "symptom") {
      result.push(row)
      continue
    }
    const ref = row.imageRef
    if (!ref) {
      result.push({ ...row, type: "image" })
      continue
    }
    const blob = await getImageBlob(ref)
    if (blob) {
      const dataUrl = await blobToDataUrl(blob)
      result.push({
        id: row.id,
        timestamp: row.timestamp,
        type: "image",
        filename: row.filename,
        imageRef: ref,
        image: dataUrl,
        ...(row.metadata ? { metadata: row.metadata } : {}),
      })
    } else {
      result.push({
        id: row.id,
        timestamp: row.timestamp,
        type: "image",
        filename: row.filename,
        imageRef: ref,
        ...(row.metadata ? { metadata: row.metadata } : {}),
      })
    }
  }
  return result
}
