import { useSkinTrack } from "@/components/skintrack-provider"

export function useProfile() {
  const { profile, setProfile, loading } = useSkinTrack()
  return { profile, setProfile, loading }
}
