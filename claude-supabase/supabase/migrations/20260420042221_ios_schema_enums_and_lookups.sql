-- Enums
do $$ begin
  create type body_view_enum as enum ('front','back');
exception when duplicate_object then null;
end $$;

do $$ begin
  create type side_enum as enum ('left','right','midline','unknown');
exception when duplicate_object then null;
end $$;

do $$ begin
  create type med_category_enum as enum ('topical','oral','injection','otc','moisturizer','cleanser','diet','environmental','avoidance','home_remedy','other');
exception when duplicate_object then null;
end $$;

do $$ begin
  create type image_kind_enum as enum ('raw','processed','mask','overlay');
exception when duplicate_object then null;
end $$;

do $$ begin
  create type processing_status_enum as enum ('pending','processing','succeeded','failed');
exception when duplicate_object then null;
end $$;

do $$ begin
  create type scale_mode_enum as enum ('aruco','fallback','manual');
exception when duplicate_object then null;
end $$;

do $$ begin
  create type segmentation_mode_enum as enum ('kmeans','grabcut','unet','none');
exception when duplicate_object then null;
end $$;

do $$ begin
  create type export_type_enum as enum ('pdf_summary','csv','json');
exception when duplicate_object then null;
end $$;

create table if not exists public.conditions (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  name text not null,
  display_name text not null,
  supports_imaging boolean not null default true,
  supports_body_map boolean not null default true,
  created_at timestamptz not null default now()
);

create table if not exists public.trigger_taxonomy (
  key text primary key,
  label text not null,
  value_type text not null check (value_type in ('boolean','text','number','enum')),
  created_at timestamptz not null default now()
);

insert into public.conditions (slug, name, display_name) values
  ('eczema', 'Eczema', 'Eczema / Atopic Dermatitis'),
  ('psoriasis', 'Psoriasis', 'Psoriasis'),
  ('acne', 'Acne', 'Acne Vulgaris'),
  ('rosacea', 'Rosacea', 'Rosacea'),
  ('dermatitis', 'Contact Dermatitis', 'Contact Dermatitis'),
  ('seborrheic_dermatitis', 'Seborrheic Dermatitis', 'Seborrheic Dermatitis'),
  ('vitiligo', 'Vitiligo', 'Vitiligo'),
  ('hives', 'Hives', 'Urticaria / Hives'),
  ('fungal', 'Fungal Infection', 'Fungal Skin Infection'),
  ('other', 'Other', 'Other Skin Condition')
on conflict (slug) do nothing;

insert into public.trigger_taxonomy (key, label, value_type) values
  ('stress', 'Stress', 'number'),
  ('sweating', 'Sweating', 'boolean'),
  ('new_detergent', 'New Detergent', 'boolean'),
  ('gluten', 'Gluten', 'boolean'),
  ('dairy', 'Dairy', 'boolean'),
  ('alcohol', 'Alcohol', 'boolean'),
  ('pollen', 'Pollen', 'boolean'),
  ('dust', 'Dust', 'boolean'),
  ('pet_dander', 'Pet Dander', 'boolean'),
  ('new_skincare', 'New Skincare Product', 'boolean'),
  ('sun_exposure', 'Sun Exposure', 'boolean'),
  ('cold_weather', 'Cold Weather', 'boolean'),
  ('hot_shower', 'Hot Shower/Bath', 'boolean'),
  ('exercise', 'Exercise', 'boolean'),
  ('lack_of_sleep', 'Lack of Sleep', 'boolean'),
  ('hormonal', 'Hormonal Changes', 'boolean'),
  ('friction', 'Friction/Rubbing', 'boolean'),
  ('other', 'Other', 'text')
on conflict (key) do nothing;

alter table public.conditions enable row level security;
drop policy if exists "conditions_read" on public.conditions;
create policy "conditions_read" on public.conditions for select to authenticated using (true);

alter table public.trigger_taxonomy enable row level security;
drop policy if exists "trigger_taxonomy_read" on public.trigger_taxonomy;
create policy "trigger_taxonomy_read" on public.trigger_taxonomy for select to authenticated using (true);
