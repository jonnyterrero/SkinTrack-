import { useSkinTrack } from "@/components/skintrack-provider"

export function useRecords() {
  const { records, loading, storageError, clearStorageError, saveRecord, replaceRecords, refresh, importBundle } =
    useSkinTrack()
  return {
    records,
    loading,
    storageError,
    clearStorageError,
    saveRecord,
    replaceRecords,
    refresh,
    importBundle,
  }
}
