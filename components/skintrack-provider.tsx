"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import { createLocalSkinTrackRepository } from "@/lib/data/local-repository"
import type { SkinTrackRepository } from "@/lib/data/repository"
import {
  emptyUserProfile,
  type Lesion,
  type NewSkinTrackRecordInput,
  type SkinTrackRecord,
  type UserProfile,
} from "@/lib/types"

type SkinTrackContextValue = {
  records: SkinTrackRecord[]
  lesions: Lesion[]
  profile: UserProfile
  loading: boolean
  storageError: string | null
  clearStorageError: () => void
  refresh: () => Promise<void>
  saveRecord: (input: NewSkinTrackRecordInput) => Promise<boolean>
  setProfile: (p: UserProfile) => void
  upsertLesion: (lesion: Lesion) => void
  replaceRecords: (records: SkinTrackRecord[]) => Promise<boolean>
  importBundle: (raw: unknown) => Promise<{ ok: true } | { ok: false; error: string }>
  repository: SkinTrackRepository
}

const SkinTrackContext = createContext<SkinTrackContextValue | null>(null)

export function SkinTrackProvider({ children }: { children: ReactNode }) {
  const repository = useMemo(() => createLocalSkinTrackRepository(), [])

  const [records, setRecords] = useState<SkinTrackRecord[]>([])
  const [lesions, setLesionsState] = useState<Lesion[]>([])
  const [profile, setProfileState] = useState<UserProfile>(emptyUserProfile())
  const [loading, setLoading] = useState(true)
  const [storageError, setStorageError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setStorageError(null)
    try {
      const [r, p] = await Promise.all([
        repository.loadRecords(),
        Promise.resolve(repository.getProfile()),
      ])
      setRecords(r)
      setProfileState(p)
      setLesionsState(repository.getLesions())
    } catch (e) {
      setStorageError((e as Error)?.message ?? "Failed to load data.")
    } finally {
      setLoading(false)
    }
  }, [repository])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const clearStorageError = useCallback(() => setStorageError(null), [])

  const saveRecord = useCallback(
    async (input: NewSkinTrackRecordInput) => {
      const result = await repository.saveRecord(input)
      if (!result.ok) {
        setStorageError(result.error)
        return false
      }
      setRecords((prev) => [result.record, ...prev])
      return true
    },
    [repository],
  )

  const setProfile = useCallback(
    (p: UserProfile) => {
      repository.setProfile(p)
      setProfileState(p)
    },
    [repository],
  )

  const upsertLesion = useCallback(
    (lesion: Lesion) => {
      repository.upsertLesion(lesion)
      setLesionsState(repository.getLesions())
    },
    [repository],
  )

  const replaceRecords = useCallback(
    async (next: SkinTrackRecord[]) => {
      const result = await repository.replaceAllRecords(next)
      if (!result.ok) {
        setStorageError(result.error)
        return false
      }
      const hydrated = await repository.loadRecords()
      setRecords(hydrated)
      return true
    },
    [repository],
  )

  const importBundle = useCallback(
    async (raw: unknown) => {
      const result = await repository.importBundle(raw, records)
      if (!result.ok) {
        return { ok: false as const, error: result.error }
      }
      setRecords(result.records)
      setProfileState(result.profile)
      setLesionsState(repository.getLesions())
      return { ok: true as const }
    },
    [repository, records],
  )

  const value = useMemo<SkinTrackContextValue>(
    () => ({
      records,
      lesions,
      profile,
      loading,
      storageError,
      clearStorageError,
      refresh,
      saveRecord,
      setProfile,
      upsertLesion,
      replaceRecords,
      importBundle,
      repository,
    }),
    [
      records,
      lesions,
      profile,
      loading,
      storageError,
      clearStorageError,
      refresh,
      saveRecord,
      setProfile,
      upsertLesion,
      replaceRecords,
      importBundle,
      repository,
    ],
  )

  return <SkinTrackContext.Provider value={value}>{children}</SkinTrackContext.Provider>
}

export function useSkinTrack(): SkinTrackContextValue {
  const ctx = useContext(SkinTrackContext)
  if (!ctx) {
    throw new Error("useSkinTrack must be used within SkinTrackProvider")
  }
  return ctx
}
