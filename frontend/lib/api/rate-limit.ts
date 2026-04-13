import type { SupabaseClient } from "@supabase/supabase-js"

/**
 * Persistent rate limit check via the Postgres `check_rate_limit` RPC.
 *
 * Defaults: 100 requests per 60 seconds per user. The function uses
 * SELECT ... FOR UPDATE inside the RPC so concurrent requests are
 * serialized correctly under contention.
 *
 * Fail-open on infra errors: if the RPC itself errors (e.g. transient
 * DB blip) we let the request through rather than locking the user out.
 */
export async function checkRateLimit(
  supabase: SupabaseClient,
  userId: string,
): Promise<boolean> {
  const { data, error } = await supabase.rpc("check_rate_limit", {
    p_user_id: userId,
  })
  if (error) return true
  return data === true
}
