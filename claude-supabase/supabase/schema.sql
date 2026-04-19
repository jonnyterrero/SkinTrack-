-- SkinTrack+ — Supabase schema (Postgres).
-- Apply in Supabase SQL editor after project creation.
--
-- Medications live in profiles.skintrack_profile JSON; v3 export adds optional `lesions` array.
-- Normalized `public.lesions` + `public.skin_events` mirror the local-first skin-event slice for analytics / hybrid sync.

create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  skintrack_profile jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.records (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  ts timestamptz not null default now(),
  record_type text not null check (record_type in ('symptom', 'image')),
  payload jsonb not null default '{}'::jsonb,
  image_storage_path text,
  created_at timestamptz not null default now()
);

create index if not exists records_user_ts_idx on public.records (user_id, ts desc);
create index if not exists records_type_idx on public.records (record_type);

create table if not exists public.lesions (
  id uuid primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  label text not null,
  created_at timestamptz not null default now(),
  archived_at timestamptz
);

create index if not exists lesions_user_idx on public.lesions (user_id);

create table if not exists public.skin_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  lesion_id uuid not null references public.lesions (id) on delete cascade,
  client_numeric_id bigint,
  ts timestamptz not null,
  severity_0_4 smallint not null check (severity_0_4 between 0 and 4),
  location_id text not null,
  itch smallint not null check (itch between 0 and 10),
  pain smallint not null check (pain between 0 and 10),
  burning smallint not null check (burning between 0 and 10),
  dryness smallint not null check (dryness between 0 and 10),
  stress smallint not null check (stress between 0 and 10),
  sleep_hours numeric not null,
  sleep_quality smallint not null check (sleep_quality between 1 and 5),
  metrics_schema_version smallint not null default 1,
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists skin_events_user_ts_idx on public.skin_events (user_id, ts desc);
create index if not exists skin_events_lesion_ts_idx on public.skin_events (lesion_id, ts desc);

alter table public.profiles enable row level security;
alter table public.records enable row level security;

create policy "profiles_select_own" on public.profiles for select using (auth.uid() = id);
create policy "profiles_insert_own" on public.profiles for insert with check (auth.uid() = id);
create policy "profiles_update_own" on public.profiles for update using (auth.uid() = id);

create policy "records_all_own" on public.records for all using (auth.uid() = user_id);

alter table public.lesions enable row level security;
alter table public.skin_events enable row level security;

create policy "lesions_all_own" on public.lesions for all using (auth.uid() = user_id);
create policy "skin_events_all_own" on public.skin_events for all using (auth.uid() = user_id);

-- Triggers ---------------------------------------------------------------

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = ''
as $$
begin
  insert into public.profiles (id)
  values (new.id)
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

create or replace function public.set_updated_at()
returns trigger language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
  before update on public.profiles
  for each row execute function public.set_updated_at();

-- Storage ----------------------------------------------------------------

insert into storage.buckets (id, name, public)
values ('skintrack-images', 'skintrack-images', false)
on conflict (id) do nothing;

create policy "skintrack_images_insert_own"
on storage.objects for insert to authenticated
with check (bucket_id = 'skintrack-images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "skintrack_images_select_own"
on storage.objects for select to authenticated
using (bucket_id = 'skintrack-images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "skintrack_images_delete_own"
on storage.objects for delete to authenticated
using (bucket_id = 'skintrack-images' and (storage.foldername(name))[1] = auth.uid()::text);

-- API keys -------------------------------------------------------------

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

-- Rate limiting --------------------------------------------------------

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
  select r.count, r.window_start
    into v_count, v_start
    from public.api_rate_limits r
    where r.user_id = p_user_id
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
