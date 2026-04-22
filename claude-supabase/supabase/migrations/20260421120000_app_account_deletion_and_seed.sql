-- Extend handle_new_user to seed app_preferences + profile email.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do update set email = excluded.email;

  insert into public.app_preferences (user_id)
  values (new.id)
  on conflict (user_id) do nothing;

  return new;
end;
$$;

-- delete_user_account: hard-delete the auth.users row for the calling user.
-- All user-owned tables cascade via on delete cascade.
-- Storage object cleanup is done by the API route before calling this RPC.
create or replace function public.delete_user_account()
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then
    raise exception 'not authenticated';
  end if;
  delete from auth.users where id = uid;
end;
$$;

grant execute on function public.delete_user_account() to authenticated;
