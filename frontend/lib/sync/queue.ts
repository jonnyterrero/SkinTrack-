import Dexie from "dexie"

export type SyncOperation = {
  id?: number
  table: "records" | "lesions" | "skin_events" | "profiles"
  action: "upsert" | "delete"
  row_id: string
  payload: Record<string, unknown>
  created_at: string
  retries: number
}

class SyncQueueDB extends Dexie {
  operations!: Dexie.Table<SyncOperation, number>

  constructor() {
    super("skintrack-sync-queue")
    this.version(1).stores({
      operations: "++id, table, created_at",
    })
  }
}

let db: SyncQueueDB | null = null

function getDb(): SyncQueueDB {
  if (!db) db = new SyncQueueDB()
  return db
}

export async function enqueue(op: Omit<SyncOperation, "id" | "created_at" | "retries">): Promise<void> {
  await getDb().operations.add({
    ...op,
    created_at: new Date().toISOString(),
    retries: 0,
  })
}

export async function peek(limit = 50): Promise<SyncOperation[]> {
  return getDb().operations.orderBy("id").limit(limit).toArray()
}

export async function dequeue(id: number): Promise<void> {
  await getDb().operations.delete(id)
}

export async function incrementRetry(id: number): Promise<void> {
  const op = await getDb().operations.get(id)
  if (op) {
    await getDb().operations.update(id, { retries: op.retries + 1 })
  }
}

export async function queueSize(): Promise<number> {
  return getDb().operations.count()
}

export async function clearQueue(): Promise<void> {
  await getDb().operations.clear()
}
