-- SkinTrack+ — Supabase schema (Postgres).
-- Apply in Supabase SQL editor after project creation.
--
-- Medications and lesion detail live in profiles.skintrack_profile JSON (export bundle v2)
-- and in record payloads — there are no separate lesions / med_schedule tables.

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

alter table public.profiles enable row level security;
alter table public.records enable row level security;

create policy "profiles_select_own" on public.profiles for select using (auth.uid() = id);
create policy "profiles_insert_own" on public.profiles for insert with check (auth.uid() = id);
create policy "profiles_update_own" on public.profiles for update using (auth.uid() = id);

create policy "records_all_own" on public.records for all using (auth.uid() = user_id);

-- Storage bucket: create in Supabase UI as "skintrack-images" (private).
-- Path convention: {user_id}/{local_numeric_record_id}/{filename}
-- Enable policies in Dashboard > Storage, or uncomment and adapt:

-- create policy "skintrack_images_insert_own"
-- on storage.objects for insert to authenticated
-- with check (bucket_id = 'skintrack-images' and (storage.foldername(name))[1] = auth.uid()::text);

-- create policy "skintrack_images_select_own"
-- on storage.objects for select to authenticated
-- using (bucket_id = 'skintrack-images' and (storage.foldername(name))[1] = auth.uid()::text);
