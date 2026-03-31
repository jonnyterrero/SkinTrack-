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

-- Storage bucket: create in Supabase UI as "skintrack-images" (private).
-- Path convention: {user_id}/{local_numeric_record_id}/{filename}
-- Enable policies in Dashboard > Storage, or uncomment and adapt:

-- create policy "skintrack_images_insert_own"
-- on storage.objects for insert to authenticated
-- with check (bucket_id = 'skintrack-images' and (storage.foldername(name))[1] = auth.uid()::text);

-- create policy "skintrack_images_select_own"
-- on storage.objects for select to authenticated
-- using (bucket_id = 'skintrack-images' and (storage.foldername(name))[1] = auth.uid()::text);
