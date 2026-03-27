import Dexie, { type EntityTable } from "dexie"

export type ImageBlobRow = {
  id: string
  blob: Blob
}

export class SkinTrackDexie extends Dexie {
  imageBlobs!: EntityTable<ImageBlobRow, "id">

  constructor() {
    super("skintrack-db")
    this.version(1).stores({
      imageBlobs: "id",
    })
  }
}

let dbSingleton: SkinTrackDexie | null = null

export function getSkinTrackDb(): SkinTrackDexie {
  if (typeof indexedDB === "undefined") {
    throw new Error("IndexedDB is not available")
  }
  if (!dbSingleton) {
    dbSingleton = new SkinTrackDexie()
  }
  return dbSingleton
}

export async function putImageBlob(id: string, blob: Blob): Promise<void> {
  await getSkinTrackDb().imageBlobs.put({ id, blob })
}

export async function getImageBlob(id: string): Promise<Blob | undefined> {
  const row = await getSkinTrackDb().imageBlobs.get(id)
  return row?.blob
}

export async function deleteImageBlob(id: string): Promise<void> {
  await getSkinTrackDb().imageBlobs.delete(id)
}
