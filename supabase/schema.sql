-- SkinTrack+ — proposed Supabase schema (Postgres).
-- Apply in Supabase SQL editor after project creation. RLS policies must be tightened per product requirements.

-- Extensions (usually enabled on Supabase)
-- create extension if not exists "uuid-ossp";

create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  skintrack_profile jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.lesions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  label text not null,
  condition text,
  body_map_area text,
  created_at timestamptz not null default now()
);

create index if not exists lesions_user_id_idx on public.lesions (user_id);

create table if not exists public.records (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  lesion_id uuid references public.lesions (id) on delete set null,
  ts timestamptz not null default now(),
  record_type text not null check (record_type in ('symptom', 'image')),
  payload jsonb not null default '{}'::jsonb,
  image_storage_path text,
  created_at timestamptz not null default now()
);

create index if not exists records_user_ts_idx on public.records (user_id, ts desc);

create table if not exists public.med_schedule (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  lesion_id uuid references public.lesions (id) on delete cascade,
  name text not null,
  dose text,
  morning boolean not null default false,
  afternoon boolean not null default false,
  evening boolean not null default false,
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists med_schedule_user_idx on public.med_schedule (user_id);

-- Row Level Security (sketch — enable and refine in Supabase dashboard)
alter table public.profiles enable row level security;
alter table public.lesions enable row level security;
alter table public.records enable row level security;
alter table public.med_schedule enable row level security;

-- Example policies (uncomment and adjust)
-- create policy "profiles_own" on public.profiles for all using (auth.uid() = id);
-- create policy "lesions_own" on public.lesions for all using (auth.uid() = user_id);
-- create policy "records_own" on public.records for all using (auth.uid() = user_id);
-- create policy "med_schedule_own" on public.med_schedule for all using (auth.uid() = user_id);

-- Storage bucket (create in Supabase UI): skintrack-images
-- Path convention: {user_id}/{record_id}/{filename}
