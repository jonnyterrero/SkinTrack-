create table if not exists public.event_images (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  skin_event_id uuid not null references public.skin_events(id) on delete cascade,
  storage_path text not null,
  kind image_kind_enum not null default 'raw',
  width int,
  height int,
  mime_type text,
  captured_at timestamptz,
  processing_status processing_status_enum not null default 'pending',
  failure_reason text
);

create index if not exists event_images_event_idx on public.event_images (skin_event_id);
alter table public.event_images enable row level security;
drop policy if exists "event_images_all_own" on public.event_images;
create policy "event_images_all_own" on public.event_images for all using (auth.uid() = user_id);

create table if not exists public.event_metrics (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  skin_event_id uuid not null references public.skin_events(id) on delete cascade,
  metrics_schema_version smallint not null,
  area_cm2 numeric,
  redness_index numeric,
  border_irregularity numeric,
  asymmetry numeric,
  delta_e numeric,
  raw_area_px numeric,
  raw_perimeter_px numeric,
  cm_per_px numeric,
  scale_mode scale_mode_enum not null default 'fallback',
  segmentation_mode segmentation_mode_enum not null default 'none',
  calibration_applied boolean not null default false,
  confidence_score numeric,
  unique (skin_event_id)
);

alter table public.event_metrics enable row level security;
drop policy if exists "event_metrics_all_own" on public.event_metrics;
create policy "event_metrics_all_own" on public.event_metrics for all using (auth.uid() = user_id);

create table if not exists public.exports (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  lesion_id uuid references public.lesions(id) on delete set null,
  export_type export_type_enum not null,
  storage_path text not null,
  start_ts timestamptz,
  end_ts timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists exports_user_idx on public.exports (user_id);
alter table public.exports enable row level security;
drop policy if exists "exports_all_own" on public.exports;
create policy "exports_all_own" on public.exports for all using (auth.uid() = user_id);
