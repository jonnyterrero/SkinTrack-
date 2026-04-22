alter table public.profiles
  add column if not exists email text,
  add column if not exists onboarding_completed_at timestamptz,
  add column if not exists consent_acknowledged_at timestamptz,
  add column if not exists consent_version text,
  add column if not exists symptom_scale_version text,
  add column if not exists clinic_notes text;

create table if not exists public.app_preferences (
  user_id uuid primary key references auth.users(id) on delete cascade,
  completed_onboarding boolean not null default false,
  preferred_log_time time,
  reminders_enabled boolean not null default true,
  quiet_hours_start time,
  quiet_hours_end time,
  units text not null default 'metric',
  theme text not null default 'system',
  consent_version text not null default '1.0',
  privacy_policy_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.app_preferences enable row level security;
drop policy if exists "app_preferences_all_own" on public.app_preferences;
create policy "app_preferences_all_own" on public.app_preferences for all using (auth.uid() = user_id);

drop trigger if exists app_preferences_set_updated_at on public.app_preferences;
create trigger app_preferences_set_updated_at
  before update on public.app_preferences
  for each row execute function public.set_updated_at();

create table if not exists public.user_conditions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  condition_id uuid not null references public.conditions(id),
  source text not null check (source in ('self_reported','clinician_diagnosed')),
  created_at timestamptz not null default now(),
  unique (user_id, condition_id)
);

create index if not exists user_conditions_user_idx on public.user_conditions (user_id);
alter table public.user_conditions enable row level security;
drop policy if exists "user_conditions_all_own" on public.user_conditions;
create policy "user_conditions_all_own" on public.user_conditions for all using (auth.uid() = user_id);

create table if not exists public.user_allergies (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  allergen text not null,
  severity text,
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists user_allergies_user_idx on public.user_allergies (user_id);
alter table public.user_allergies enable row level security;
drop policy if exists "user_allergies_all_own" on public.user_allergies;
create policy "user_allergies_all_own" on public.user_allergies for all using (auth.uid() = user_id);
