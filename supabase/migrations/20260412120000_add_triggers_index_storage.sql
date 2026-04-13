-- Phase 1 completion: triggers, missing index, storage bucket + RLS.

-- 1. records(record_type) index
create index if not exists records_type_idx on public.records (record_type);

-- 2. handle_new_user trigger: auto-create a profiles row on sign-up
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

-- 3. updated_at trigger on profiles
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

-- 4. Storage bucket + RLS policies
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
