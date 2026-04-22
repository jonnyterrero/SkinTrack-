import { createClient, type SupabaseClient } from "@supabase/supabase-js"

let cached: SupabaseClient | null = null

/**
 * Server-only Supabase client using the service role key.
 * NEVER import this into a client component or export a route
 * that echoes its credentials back to the browser.
 */
export function getSupabaseAdminClient(): SupabaseClient {
  if (cached) return cached
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!url || !serviceKey) {
    throw new Error(
      "Missing SUPABASE_SERVICE_ROLE_KEY or NEXT_PUBLIC_SUPABASE_URL for admin client.",
    )
  }
  cached = createClient(url, serviceKey, {
    auth: { autoRefreshToken: false, persistSession: false },
  })
  return cached
}
