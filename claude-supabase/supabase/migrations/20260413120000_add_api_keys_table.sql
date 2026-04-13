-- Phase 8: dedicated api_keys table.
-- Replaces the temporary JSONB array at profiles.skintrack_profile.apiKeys.

create table if not exists public.api_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  hash text not null unique,
  prefix text not null,
  created_at timestamptz not null default now(),
  last_used_at timestamptz,
  revoked_at timestamptz
);

create index if not exists api_keys_user_idx on public.api_keys (user_id);
create index if not exists api_keys_hash_idx on public.api_keys (hash);

alter table public.api_keys enable row level security;

drop policy if exists "api_keys_all_own" on public.api_keys;
create policy "api_keys_all_own" on public.api_keys
  for all using (auth.uid() = user_id);
