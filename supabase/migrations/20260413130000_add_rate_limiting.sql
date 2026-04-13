-- Phase 8: persistent Postgres-backed rate limiting.
-- Replaces the in-memory token bucket that resets on serverless cold start.

create table if not exists public.api_rate_limits (
  user_id uuid primary key references auth.users (id) on delete cascade,
  count integer not null default 0,
  window_start timestamptz not null default now()
);

alter table public.api_rate_limits enable row level security;
-- No client-facing policies: only callable via the SECURITY DEFINER function below.

create or replace function public.check_rate_limit(
  p_user_id uuid,
  p_max int default 100,
  p_window_seconds int default 60
)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_count int;
  v_start timestamptz;
begin
  select count, window_start
    into v_count, v_start
    from public.api_rate_limits
    where user_id = p_user_id
    for update;

  if not found then
    insert into public.api_rate_limits (user_id, count, window_start)
      values (p_user_id, 1, now());
    return true;
  end if;

  if now() - v_start > make_interval(secs => p_window_seconds) then
    update public.api_rate_limits
      set count = 1, window_start = now()
      where user_id = p_user_id;
    return true;
  end if;

  if v_count >= p_max then
    return false;
  end if;

  update public.api_rate_limits
    set count = count + 1
    where user_id = p_user_id;
  return true;
end;
$$;

grant execute on function public.check_rate_limit(uuid, int, int) to authenticated;
