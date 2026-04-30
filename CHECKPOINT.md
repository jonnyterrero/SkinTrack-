# SkinTrack+ — Checkpoint (2026-04-22)

## Where we are

Branch `Skintrack+-backend` on `origin` (GitHub `jonnyterrero/SkinTrack-`).

Latest commits (newest first):

```
7da82c2  feat(sprint-2-3): finish iOS schema alignment, build green
17bd981  wip: align sprint 1-3 backend to real supabase schema
5d5e4bf  fix(deploy): copy vercel.json to frontend/ for root directory deploy
```

Both `7da82c2` and `17bd981` are pushed and present on the remote — verified with `git fetch && git log origin/Skintrack+-backend`.

Build status: `tsc --noEmit` clean, `next build --webpack` succeeds with 39 routes.

## What was done across the past two sessions

### Session 1 (large WIP — commit `17bd981`)

- Pulled the live `ios_schema_*` migrations out of Supabase (project `lxispkvlxcvwdrxhxegs`) into `claude-supabase/supabase/migrations/`. Five files now mirror prod:
  - `20260420042221_ios_schema_enums_and_lookups.sql`
  - `20260420042300_ios_schema_profile_preferences_allergies.sql`
  - `20260420042331_ios_schema_meds_locations_triggers.sql`
  - `20260420042354_ios_schema_images_metrics_exports.sql`
  - `20260421120000_app_account_deletion_and_seed.sql` (new — extends `handle_new_user()`, adds `delete_user_account()` RPC)
- Rewrote `frontend/lib/types/backend.ts` to mirror the live DB (real enums, normalized tables, trimmed `app_preferences`).
- Rewrote / added validators in `frontend/lib/validators/`:
  - `medications` (with `MED_CATEGORIES`)
  - `event-context` (triggers as plain text against taxonomy; products use `MED_CATEGORIES`)
  - `event-assets` (image kinds without thumbnail; processing status = `succeeded`; export type = `pdf_summary | csv | json`; `storage_path` required)
  - `lesion-locations` (sides = `left | right | midline | unknown`; added `scalp` region)
  - `profile` (added email, clinic notes, onboarding/consent stamps)
  - `app-preferences` (only real columns)
  - `user-conditions`, `user-allergies` (new)
- Added 15 new API routes under `frontend/app/api/`: `account` (GET dump + DELETE), `app-preferences`, `conditions`, `user-conditions`, `user-allergies`, `triggers`, `medications`, `lesion-locations`, `event-medications`, `event-triggers`, `event-products`, `event-images`, `event-metrics`, `exports`.
- Added/rewrote pages:
  - `/onboarding` — 5-step setup, persists conditions/allergies via separate endpoints
  - `/settings` — profile + clinic notes, allergies CRUD, reminder prefs, export/delete account
  - `/medications` — `med_catalog` CRUD
  - `/checkin` — meds toggle, triggers (from taxonomy), products
  - `/body-map` — front/back silhouette pin component
  - `/export`, `/export/summary` — CSV + JSON download + printable clinician report
  - `/about`, `/legal/privacy`, `/legal/terms`, `/support`
- Added `frontend/components/body-map.tsx`, `frontend/hooks/useReminders.ts`, `frontend/lib/api/client.ts`, `frontend/lib/supabase/admin-client.ts`.

### Session 2 (build green — commit `7da82c2`)

- `app/checkin/page.tsx` — drops hardcoded TRIGGER_KEYS / PRODUCT_TYPES; pulls taxonomy from `/api/triggers` at runtime; product types use `MedCategory`.
- `app/export/page.tsx` — removes broken `/api/exports` POST writes (server requires `storage_path NOT NULL`, client doesn't have one); pure client download for CSV/JSON.
- `app/export/summary/page.tsx` — reads `/api/user-conditions`, `/api/user-allergies`, and `profile.clinic_notes` instead of removed `prefs.allergies` / `prefs.dermatologist_notes` / `prefs.primary_condition_id`.
- `app/body-map/page.tsx` — types `body_region` as `BodyRegion` when mapping `LesionLocation`; tightens `BodyPin.side` to non-null in `components/body-map.tsx`.
- `app/medications/page.tsx` — exposes `otc` and `avoidance` categories.
- `hooks/useReminders.ts` — drops references to removed `prefs.reminder_med_enabled` / `reminder_photo_enabled`.
- `app/settings/page.tsx` — null-guards `apiSend<UserAllergy>` result.
- Wrapped `useSearchParams` pages (`body-map`, `checkin`, `export/summary`) in `<Suspense>` so they prerender under Next 16.
- Updated `CLAUDE.md` Current State Audit, added Phase 8 / 9 / 10 entries.

## What still blocks ship

| Item | Notes |
|---|---|
| **Capacitor wrap (iOS/Android)** | Not started. Need `capacitor.config.ts`, iOS + Android project init, splash + icon assets, native runtime smoke test. |
| **Turbopack build** | Broken on this monorepo on Windows ("Next.js inferred your workspace root, but it may not be correct"). Must use `next build --webpack` until upstream fix. Vercel deploy uses `frontend/` as root and works. |
| **`/api/exports` endpoint** | Schema requires `storage_path NOT NULL`, but the export UI only does client downloads. Needs server-side export generation (write JSON/CSV/PDF to `skintrack-images` bucket, then row insert). |
| **`/api/keys` storage** | Keys still stored in `profiles.skintrack_profile` JSONB. No persistent api_keys table. |
| **Pull/restore from Supabase → local** | Sync engine is push-only. |
| **Rate limiter** | In-memory token bucket; resets on serverless cold start. |
| **Automated tests** | None. |
| **PWA polish** | Manifest + icons exist but service worker `/sw.js` and update flow have not been re-verified after these changes. |

## How to resume

```bash
# from repo root
git checkout Skintrack+-backend
git pull origin Skintrack+-backend

cd frontend
../node_modules/.bin/tsc --noEmit       # should be silent
../node_modules/.bin/next build --webpack # should produce 39 routes

# dev server
../node_modules/.bin/next dev
```

`.env.local` (in `frontend/`) needs:

```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
NEXT_PUBLIC_ENABLE_UNET=false
```

## Suggested next step (Capacitor wrap)

Already offered to schedule a one-off agent for ~one week out to scaffold:
- `frontend/capacitor.config.ts`
- `npx cap add ios` and `npx cap add android` from `frontend/`
- Splash + icon assets in `public/`
- Update `next.config.mjs` for static export (`output: "export"`) or hybrid
- Smoke-test on a simulator and report back

If the schedule fires, it'll open a PR rather than push to `Skintrack+-backend` directly.

## Reference

- Project blueprint: [`CLAUDE.md`](./CLAUDE.md)
- Live Supabase project ID: `lxispkvlxcvwdrxhxegs` (us-east-2)
- Frontend root: [`frontend/`](./frontend/)
- Migrations: [`claude-supabase/supabase/migrations/`](./claude-supabase/supabase/migrations/)
