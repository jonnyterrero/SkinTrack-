create table if not exists public.lesion_locations (
  id uuid primary key default gen_random_uuid(),
  lesion_id uuid not null references public.lesions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  body_view body_view_enum not null,
  body_region text not null,
  side side_enum not null default 'unknown',
  loc_x numeric not null,
  loc_y numeric not null,
  created_at timestamptz not null default now()
);

create index if not exists lesion_locations_lesion_idx on public.lesion_locations (lesion_id, created_at desc);
create index if not exists lesion_locations_user_idx on public.lesion_locations (user_id);
alter table public.lesion_locations enable row level security;
drop policy if exists "lesion_locations_all_own" on public.lesion_locations;
create policy "lesion_locations_all_own" on public.lesion_locations for all using (auth.uid() = user_id);

create table if not exists public.med_catalog (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  category med_category_enum not null,
  dose text,
  frequency text,
  morning boolean not null default false,
  afternoon boolean not null default false,
  evening boolean not null default false,
  is_prescription boolean not null default false,
  prescribed_by text,
  start_date date,
  end_date date,
  notes text,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists med_catalog_user_active_idx on public.med_catalog (user_id, active);
alter table public.med_catalog enable row level security;
drop policy if exists "med_catalog_all_own" on public.med_catalog;
create policy "med_catalog_all_own" on public.med_catalog for all using (auth.uid() = user_id);

drop trigger if exists med_catalog_set_updated_at on public.med_catalog;
create trigger med_catalog_set_updated_at
  before update on public.med_catalog
  for each row execute function public.set_updated_at();

create table if not exists public.lesion_medications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  lesion_id uuid not null references public.lesions(id) on delete cascade,
  med_catalog_id uuid not null references public.med_catalog(id) on delete cascade,
  scope text not null check (scope in ('global','lesion_specific')),
  unique (lesion_id, med_catalog_id)
);

alter table public.lesion_medications enable row level security;
drop policy if exists "lesion_medications_all_own" on public.lesion_medications;
create policy "lesion_medications_all_own" on public.lesion_medications for all using (auth.uid() = user_id);

create table if not exists public.event_medications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  skin_event_id uuid not null references public.skin_events(id) on delete cascade,
  med_catalog_id uuid not null references public.med_catalog(id) on delete restrict,
  taken boolean not null,
  amount_text text,
  missed_reason text,
  notes text,
  unique (skin_event_id, med_catalog_id)
);

alter table public.event_medications enable row level security;
drop policy if exists "event_medications_all_own" on public.event_medications;
create policy "event_medications_all_own" on public.event_medications for all using (auth.uid() = user_id);

create table if not exists public.event_triggers (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  skin_event_id uuid not null references public.skin_events(id) on delete cascade,
  trigger_key text not null,
  trigger_value_text text,
  unique (skin_event_id, trigger_key)
);

alter table public.event_triggers enable row level security;
drop policy if exists "event_triggers_all_own" on public.event_triggers;
create policy "event_triggers_all_own" on public.event_triggers for all using (auth.uid() = user_id);

create table if not exists public.event_products (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  skin_event_id uuid not null references public.skin_events(id) on delete cascade,
  product_name text not null,
  product_type med_category_enum not null,
  first_use boolean not null default false,
  used boolean not null default true,
  perceived_benefit smallint,
  adverse_reaction boolean not null default false,
  notes text
);

alter table public.event_products enable row level security;
drop policy if exists "event_products_all_own" on public.event_products;
create policy "event_products_all_own" on public.event_products for all using (auth.uid() = user_id);
