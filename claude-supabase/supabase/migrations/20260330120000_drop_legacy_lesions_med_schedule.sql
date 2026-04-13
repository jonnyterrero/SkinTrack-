-- Optional migration: remove legacy tables if you previously applied an older schema.sql
-- that included public.lesions, public.med_schedule, and records.lesion_id.

drop table if exists public.med_schedule cascade;

alter table if exists public.records drop constraint if exists records_lesion_id_fkey;

alter table if exists public.records drop column if exists lesion_id;

drop table if exists public.lesions cascade;
