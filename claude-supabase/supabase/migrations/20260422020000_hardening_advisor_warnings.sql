-- Hardening pass: address Supabase advisor warnings.
-- 2026-04-22.
--
-- 1. api_rate_limits has RLS on but no policy. Only check_rate_limit() (SECURITY
--    DEFINER) ever reads/writes it; an explicit deny-all policy makes that
--    intent legible and silences the linter.
-- 2. set_updated_at had a mutable search_path.
-- 3. handle_new_user is a trigger handler; nobody should be able to call it via
--    PostgREST. Revoke from anon + authenticated (postgres still has it).
-- 4. check_rate_limit and delete_user_account are intentionally callable from
--    PostgREST, but only by authenticated users. Revoke from anon.

-- 1. Explicit deny-all on api_rate_limits.
drop policy if exists "api_rate_limits_no_direct_access" on public.api_rate_limits;
create policy "api_rate_limits_no_direct_access"
  on public.api_rate_limits
  for all
  using (false)
  with check (false);

-- 2. Pin search_path on set_updated_at.
alter function public.set_updated_at() set search_path = public, pg_temp;

-- 3. Lock down handle_new_user (trigger handler — never RPC-callable).
revoke execute on function public.handle_new_user() from anon, authenticated, public;

-- 4. Lock down RPCs that are SECURITY DEFINER but should require auth.
revoke execute on function public.check_rate_limit(uuid, integer, integer) from anon, public;
revoke execute on function public.delete_user_account() from anon, public;

-- Re-grant to authenticated explicitly so signed-in users can still call them.
grant execute on function public.check_rate_limit(uuid, integer, integer) to authenticated;
grant execute on function public.delete_user_account() to authenticated;
