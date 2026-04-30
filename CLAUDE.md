# CLAUDE.md — SkinTrack+ Backend Architecture Blueprint

---

# 1. PROJECT OVERVIEW & ARCHITECTURE DECISION RECORD

## What SkinTrack+ is

SkinTrack+ is a **local-first Progressive Web App (PWA)** designed for users managing chronic skin conditions. It enables:

* Tracking lesions over time
* Logging symptoms (itch, pain, stress, etc.)
* Capturing and analyzing skin images
* Identifying correlations between behavior and flare-ups

## Scope

* **NOT a diagnostic tool**
* **User-controlled health tracking system**
* Designed for:

  * Longitudinal self-monitoring
  * Pattern recognition
  * Data sharing with healthcare providers (future)

## Core Design Philosophy

* **Local-first → cloud-augmented**
* Offline capability is **non-negotiable**
* Cloud is used for:

  * Backup
  * Cross-device sync
  * Analytics
  * Future ML

---

## Architecture Decision

### Chosen Stack

* Frontend: Next.js (App Router)
* Local Persistence:

  * localStorage (structured data)
  * IndexedDB (images via Dexie)
* Backend: Supabase

  * Auth
  * Postgres
  * Storage
* Sync Layer: Custom engine

### Why Supabase

* Native Postgres (schema control)
* Built-in Auth + RLS
* Storage integration
* Works with local-first model
* Avoids Firebase vendor lock-in

### Repository layout

* **Next.js (App Router)** — `frontend/` (`package.json`, `app/`, `lib/`, `middleware.ts`, etc.).
* **Supabase SQL & migrations** — `claude-supabase/supabase/`.
* **Ancillary backend notes** — `backend/skintracker+/`.
* **`cursor/`**, **`legacy/`** — stubs for IDE notes and archived material.

Unless noted otherwise, file paths in this document are relative to the repository root.

---

## Data Flow (Final)

```
[ Browser UI ]
     ↓
[ Local Repository ]
     ↓
[ Sync Engine ]
     ↓
[ Supabase Auth ] → [ Postgres DB ]
                      ↓
                  [ Storage ]
```

---

# 2. CURRENT STATE AUDIT

> Last updated: 2026-04-22 — Sprint 1–3 (iOS schema) types/validators/routes/UI aligned to live DB. `npx tsc --noEmit` and `next build --webpack` pass.

## Persistence

* Local-first via:

  * `localStorage` (records, profile, metadata)
  * IndexedDB (image blobs via Dexie)
* Repository:

  * `createLocalSkinTrackRepository()` = production path (`frontend/lib/data/local-repository.ts`)
  * `createSupabaseSkinTrackRepository()` = stub only, not wired (`frontend/lib/data/supabase-repository.ts`)

## Sync

* Legacy push-only: `syncLocalBundleToSupabase()` in `frontend/lib/data/sync.ts` (full replace, still present but no longer called from UI)
* New sync engine: `frontend/lib/sync/engine.ts` + `frontend/lib/sync/queue.ts` — idempotent upserts via IndexedDB queue
* Hook: `frontend/hooks/useSyncEngine.ts` — auto-syncs on 30 s interval when authenticated
* **Integrated**: `SkinTrackProvider` calls `enqueue()` on every write + runs `useSyncEngine()` for auto-sync
* **Integrations tab**: replaced legacy `syncLocalBundleToSupabase` with "Sync now" button using new engine
* API: `POST /api/sync` — accepts batch operations from the client queue

## API

Sprint 1 (auth/sync foundation):
* `/api/skintrack` = deprecated echo stub (`frontend/app/api/skintrack/route.ts`)
* Profile: `GET /api/profile`, `PUT /api/profile`
* Records: `GET /api/records`, `POST /api/records`, `PATCH /api/records/[id]`, `DELETE /api/records/[id]`
* Lesions: `GET /api/lesions`, `POST /api/lesions`, `GET /api/lesions/[id]`, `PATCH /api/lesions/[id]`, `DELETE /api/lesions/[id]`
* Skin Events: `GET /api/skin-events`, `POST /api/skin-events`, `GET /api/skin-events/[id]`, `PATCH /api/skin-events/[id]`, `DELETE /api/skin-events/[id]`
* Upload: `POST /api/upload`, `GET /api/upload?path=` (signed URL), `DELETE /api/upload/[id]`
* Sync: `POST /api/sync`
* Webhook: `POST /api/webhook`
* API Keys: `POST /api/keys`, `DELETE /api/keys/[id]`

Sprint 2 (iOS schema — onboarding, meds, body map, triggers/products):
* App preferences: `GET/PUT /api/app-preferences`
* Conditions catalog: `GET /api/conditions` (read-only)
* User conditions: `GET /api/user-conditions`, `POST /api/user-conditions`, `DELETE /api/user-conditions/[id]`
* User allergies: `GET /api/user-allergies`, `POST /api/user-allergies`, `PATCH /api/user-allergies/[id]`, `DELETE /api/user-allergies/[id]`
* Trigger taxonomy: `GET /api/triggers` (read-only canonical list)
* Medications catalog: `GET/POST /api/medications`, `PATCH/DELETE /api/medications/[id]`
* Lesion locations: `GET/POST /api/lesion-locations`, `PATCH/DELETE /api/lesion-locations/[id]`
* Event medications: `GET/POST /api/event-medications`, `PATCH/DELETE /api/event-medications/[id]`
* Event triggers: `GET/POST /api/event-triggers`, `DELETE /api/event-triggers/[id]`
* Event products: `GET/POST /api/event-products`, `PATCH/DELETE /api/event-products/[id]`
* Event images: `GET/POST /api/event-images`, `DELETE /api/event-images/[id]`
* Event metrics: `GET/POST /api/event-metrics` (upsert on `skin_event_id`)
* Exports log: `GET/POST /api/exports`

Sprint 3 (App Store / Play Store compliance):
* Account: `GET /api/account` (full data dump for export), `DELETE /api/account` (hard delete via `delete_user_account()` RPC + storage cleanup)

## Supabase

Project `lxispkvlxcvwdrxhxegs` (us-east-2, ACTIVE_HEALTHY).

Schema (real, applied to DB — pulled into `claude-supabase/supabase/migrations/`):
* `20260420042221_ios_schema_enums_and_lookups.sql` — enums (`body_view_enum`, `side_enum`, `med_category_enum`, `image_kind_enum`, `processing_status_enum`, `scale_mode_enum`, `segmentation_mode_enum`, `export_type_enum`); lookup tables `conditions`, `trigger_taxonomy`.
* `20260420042300_ios_schema_profile_preferences_allergies.sql` — `profiles` extended (`email`, `clinic_notes`, `onboarding_completed_at`, `consent_*`, `symptom_scale_version`); new tables `app_preferences`, `user_conditions`, `user_allergies`.
* `20260420042331_ios_schema_meds_locations_triggers.sql` — `med_catalog`, `lesion_medications`, `lesion_locations`, `event_medications`, `event_triggers`, `event_products`.
* `20260420042354_ios_schema_images_metrics_exports.sql` — `event_images`, `event_metrics`, `exports`.
* `20260421120000_app_account_deletion_and_seed.sql` — extends `handle_new_user()` to seed `app_preferences` + `profiles.email`; adds `delete_user_account()` SECURITY DEFINER RPC.

Plus the legacy Sprint 1 tables (profiles base, records, lesions, skin_events) from `schema.sql` and `20260412120000_add_triggers_index_storage.sql`.

RLS enabled on every user-owned table; all `for all using (auth.uid() = user_id)`. `conditions` and `trigger_taxonomy` are world-readable.

Storage bucket `skintrack-images` (private, `{user_id}/{record_id}/{filename}`) with insert/select/delete RLS scoped to `auth.uid()`.

## Auth

* Browser client: `frontend/lib/supabase/browser-client.ts` (lazy singleton)
* Server client: `frontend/lib/supabase/server.ts` (cookie-based via `@supabase/ssr`)
* Session refresh: `frontend/middleware.ts`
* Auth callback: `frontend/app/auth/callback/route.ts` (PKCE code exchange)
* Login page: `frontend/app/login/page.tsx` (magic link form)
* Auth context: `frontend/context/AuthContext.tsx` (wired into `frontend/app/providers.tsx`)

## Security

* Rate limiting: `frontend/lib/api/rate-limit.ts` (100 req/min per user, in-memory token bucket) — **active on all API routes** via `requireAuthAndRateLimit()`
* API key hashing: `frontend/lib/api/api-keys.ts` (SHA-256, `sk_` prefixed keys)
* Input sanitization: `frontend/lib/api/sanitize.ts` (strips angle brackets from text fields) — **active on all mutating API routes** via `sanitizedBody()` helper in `frontend/lib/api/helpers.ts`
* Zod validation: `frontend/lib/validators/` — `profile`, `app-preferences`, `user-conditions`, `user-allergies`, `medications`, `event-context`, `event-assets`, `lesion-locations`, `records`, `lesions`, `skin-events`, `upload`.
* Upload validation: MIME type allowlist + magic byte verification + 10 MB limit

## App Store / Play Store compliance

* Account deletion: `DELETE /api/account` — purges storage objects under `{user_id}/`, then calls `delete_user_account()` RPC; cascades remove all DB rows.
* Data export: `GET /api/account` — returns full user dump (profile, prefs, conditions, allergies, lesions, events, meds, event_*, lesion_locations, exports). Settings page wires "Export my data (JSON)".
* Onboarding gate: `frontend/app/onboarding/page.tsx` — disclaimer ack → profile/conditions/allergies → reminder prefs → review. Stamps `profile.consent_acknowledged_at`, `profile.consent_version`, `profile.symptom_scale_version`, `profile.onboarding_completed_at`, `app_preferences.completed_onboarding`.
* Legal: `/legal/privacy`, `/legal/terms`, `/about`, `/support` static pages.
* Home redirect: signed-in users with `app_preferences.completed_onboarding=false` are pushed to `/onboarding`.

## UI surface (Sprint 2/3)

* `/onboarding` — 5-step setup, persists conditions to `user_conditions`, allergies to `user_allergies`.
* `/settings` — profile name + clinic notes, allergies CRUD, reminder time/toggle, export/delete account.
* `/medications` — `med_catalog` CRUD with `med_category_enum` + dose/frequency/morning/afternoon/evening + Rx flag.
* `/checkin?event=<id>` — pulls `trigger_taxonomy` from `/api/triggers`, lets user toggle med adherence, triggers, and log products (`med_category_enum`).
* `/body-map?lesion=<id>` — front/back silhouette, click to pin to `lesion_locations` with region + side (`left|right|midline|unknown`).
* `/export` + `/export/summary` — CSV / JSON download + printable clinician summary (uses `/api/user-conditions`, `/api/user-allergies`, `/api/profile.clinic_notes`).

## Auth UI

* Sign-in/sign-out button in `frontend/app/page.tsx` header — uses `useAuth()` from AuthContext
* Sync status cloud icon (with pending count badge) visible when signed in
* Auth is opt-in — app works fully offline without signing in (local-first preserved)

## Risks (remaining)

* Legacy `syncLocalBundleToSupabase` still present in `frontend/lib/data/sync.ts` (no longer called from UI, but not deleted).
* Rate limiter is in-memory (resets on serverless cold start).
* No persistent API key storage table (keys stored in `profiles.skintrack_profile` JSONB).
* No pull/restore from Supabase back to local (sync is push-only).
* No automated tests.
* Capacitor native wrap (iOS/Android) not yet scaffolded.
* Turbopack build is broken in this monorepo on Windows; production build must use `next build --webpack`.
* `/api/exports` requires `storage_path NOT NULL` but the export UI currently downloads client-side only (no row written). Server-side export generation is not yet implemented.

---

# 3. TARGET BACKEND ARCHITECTURE

## Hybrid Local-First Model

### Local = Source of UX Truth

* Fast writes
* Offline operation
* IndexedDB for images

### Supabase = Source of Persistence Truth

* Durable storage
* Cross-device access
* Analytics-ready

---

## Responsibilities

| Layer       | Responsibility              |
| ----------- | --------------------------- |
| Local       | Immediate writes, caching   |
| Sync Engine | Queue + conflict resolution |
| Supabase    | Durable storage             |
| Storage     | Image persistence           |

---

# 4. ENVIRONMENT & CONFIGURATION

## Required `.env.local`

```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
NEXT_PUBLIC_ENABLE_UNET=false
```

## Rules

* `NEXT_PUBLIC_*` → client-safe
* `SERVICE_ROLE_KEY` → server-only (NEVER expose)

## Setup Checklist

1. Enable Email Auth (magic link)
2. Enable PKCE
3. Add redirect URLs:

   * localhost
   * production domain

---

# 5. AUTHENTICATION SYSTEM

## Flow

1. User enters email
2. Supabase sends magic link
3. User clicks link
4. Session established via PKCE

## Components

### Required Files

* `frontend/lib/supabase/browser-client.ts` *(exists — lazy singleton browser client)*
* `frontend/lib/supabase/server.ts` *(exists — cookie-based server client via `@supabase/ssr`)*
* `frontend/middleware.ts` *(exists — session refresh middleware)*
* `frontend/app/login/page.tsx` *(exists — magic link login form)*
* `frontend/app/auth/callback/route.ts` *(exists — PKCE code exchange)*
* `frontend/context/AuthContext.tsx` *(exists — React auth context wired into providers)*

## Behavior

* Session persists via cookies
* Auto refresh handled by Supabase
* Logout clears session + local sync state

---

# 6. DATABASE SCHEMA

## Tables

### profiles

* id (uuid)
* display_name
* skintrack_profile (jsonb)
* timestamps

### records

* id (uuid)
* user_id
* ts
* record_type
* payload (jsonb)
* image_storage_path

### lesions

* id (uuid)
* label
* archived_at

### skin_events

* normalized event data

---

## Implemented (Phase 1)

### Triggers

* `on_auth_user_created` → `handle_new_user()` — auto-creates profiles row
* `profiles_set_updated_at` → `set_updated_at()` — maintains `updated_at`

### Indexes

* `records_user_ts_idx` — records(user_id, ts desc)
* `records_type_idx` — records(record_type)
* `lesions_user_idx` — lesions(user_id)
* `skin_events_user_ts_idx` — skin_events(user_id, ts desc)
* `skin_events_lesion_ts_idx` — skin_events(lesion_id, ts desc)

---

## Storage

Bucket: `skintrack-images` (private, created via SQL)

Path convention:

```
{user_id}/{record_id}/{filename}
```

RLS policies: insert, select, delete — all scoped to `auth.uid()` via folder path.

---

# 7. TYPESCRIPT TYPES

## Core Types

* Profile
* Record
* SkinEvent
* Lesion
* ImageRecord
* SyncPayload

## Validation

* Zod schemas for all API inputs

---

# 8. API ROUTES

## Profile

* GET /api/profile
* PUT /api/profile

## Records

* GET /api/records
* POST /api/records
* PATCH /api/records/[id]
* DELETE /api/records/[id]

## Lesions

* CRUD endpoints

## Skin Events

* CRUD endpoints

## Images

* POST /api/upload
* DELETE /api/upload/[id]
* GET signed URL

## Sync

* POST /api/sync

## Webhook

* POST /api/webhook

## API Keys

* POST /api/keys
* DELETE /api/keys/[id]

---

## `/api/skintrack`

* Deprecated
* Keep for informational fallback

---

# 9. SYNC ARCHITECTURE

## Current (Legacy)

* Full delete + reinsert

## Final Design

### Strategy

* Idempotent upserts
* Client ID + server UUID mapping

### State Machine

```
idle → dirty → syncing → synced | error
```

### Rules

* Server wins on conflicts
* Local queue persists in IndexedDB
* Retry on failure

---

## Files

* `frontend/lib/sync/engine.ts`
* `frontend/lib/sync/queue.ts`
* `frontend/hooks/useSyncEngine.ts`

---

# 10. IMAGE PIPELINE

## Flow

1. Capture image
2. Store locally (IndexedDB)
3. Upload to Supabase Storage
4. Store metadata in Postgres

## Validation

* MIME + magic byte
* max 10MB
* dimension limits

## Signed URLs

* Expire in 1 hour

## Thumbnails

* Generated via Edge Function

---

# 11. RLS POLICY AUDIT

## Required Policies

* profiles → own
* records → own
* lesions → own
* skin_events → own
* storage → path scoped to user_id

## Service Role

* ONLY used server-side
* NEVER exposed to browser

---

# 12. ERROR HANDLING

## Format

```
{ error, code, details }
```

## Codes

* UNAUTHORIZED
* FORBIDDEN
* VALIDATION_ERROR
* DB_ERROR
* STORAGE_ERROR
* CONFLICT

---

# 13. RATE LIMITING & SECURITY

## Required

* 100 req/min per user
* API key hashing (SHA-256)
* sanitize all text input
* validate uploads strictly
* restrict CORS

---

# 14. FILE STRUCTURE

```
frontend/app/api/*
frontend/lib/supabase/*
frontend/lib/sync/*
frontend/lib/types/*
frontend/lib/validators/*
frontend/context/*
frontend/middleware.ts
```

---

# 15. IMPLEMENTATION PHASES

## Phase 1 — DONE (2026-04-12)

* Supabase setup
* schema + RLS
* Triggers (handle_new_user, updated_at)
* Storage bucket + RLS policies
* Migration: `20260412120000_add_triggers_index_storage.sql`

## Phase 2 — DONE (2026-04-13)

* Auth system: server client, middleware, callback, login page, AuthContext
* `@supabase/ssr` added for cookie-based server auth
* `AuthProvider` wired into `frontend/app/providers.tsx`

## Phase 3 — DONE (2026-04-13)

* Zod validators: `frontend/lib/validators/` (profile, records, lesions, skin-events, upload)
* API helpers: `frontend/lib/api/helpers.ts` (auth guard, error format, DB error)
* Profile API: `GET /api/profile`, `PUT /api/profile`
* Records API: `GET/POST /api/records`, `PATCH/DELETE /api/records/[id]`

## Phase 4 — DONE (2026-04-13)

* Lesions API: `GET/POST /api/lesions`, `GET/PATCH/DELETE /api/lesions/[id]`
* Skin Events API: `GET/POST /api/skin-events`, `GET/PATCH/DELETE /api/skin-events/[id]`

## Phase 5 — DONE (2026-04-13)

* Upload API: `POST /api/upload` (multipart, MIME + magic byte validation, 10 MB limit)
* Signed URLs: `GET /api/upload?path=` (1 hr expiry)
* Delete images: `DELETE /api/upload/[id]`
* Validators: `frontend/lib/validators/upload.ts`

## Phase 6 — DONE (2026-04-13)

* Sync queue: `frontend/lib/sync/queue.ts` (IndexedDB via Dexie, enqueue/dequeue/peek)
* Sync engine: `frontend/lib/sync/engine.ts` (idempotent upserts, retry with backoff, state machine)
* Hook: `frontend/hooks/useSyncEngine.ts` (30 s auto-sync interval)
* Sync API: `POST /api/sync` (batch operations endpoint)

## Phase 7 — DONE (2026-04-13)

* Rate limiting: `frontend/lib/api/rate-limit.ts` (100 req/min in-memory token bucket)
* API key hashing: `frontend/lib/api/api-keys.ts` (SHA-256, `sk_` prefix generation)
* Input sanitization: `frontend/lib/api/sanitize.ts` (angle bracket stripping)
* Webhook: `POST /api/webhook` (forwards events to user-configured URL)
* API Keys: `POST /api/keys`, `DELETE /api/keys/[id]`

## Phase 8 (Sprint 2 — iOS schema) — DONE (2026-04-22)

* Pulled real `ios_schema_*` migrations into `claude-supabase/supabase/migrations/`.
* Backend types rewritten in `frontend/lib/types/backend.ts` to match live DB.
* Validators added: `medications`, `event-context`, `event-assets`, `lesion-locations`, `app-preferences`, `user-conditions`, `user-allergies`, plus `profile` extended.
* API routes added (15 new): `app-preferences`, `conditions`, `user-conditions`, `user-allergies`, `triggers`, `medications`, `lesion-locations`, `event-medications`, `event-triggers`, `event-products`, `event-images`, `event-metrics`, `exports`.
* UI: `/onboarding` (5-step), `/settings` (rewritten), `/medications`, `/checkin`, `/body-map`, `/export`, `/export/summary`, `/about`, `/legal/privacy`, `/legal/terms`, `/support`.
* Body map component (`frontend/components/body-map.tsx`) — front/back silhouette with region click-to-pin.

## Phase 9 (Sprint 3 — App Store / Play Store readiness) — DONE (2026-04-22)

* Account API: `GET /api/account` (full export dump), `DELETE /api/account` (storage cleanup + RPC delete).
* Migration `20260421120000_app_account_deletion_and_seed.sql`: extends `handle_new_user()` to seed `app_preferences` and stamp `profiles.email`; adds `delete_user_account()` SECURITY DEFINER RPC.
* Onboarding gate stamps consent + symptom-scale versions on `profiles`.
* Legal pages, support page, About page in place.
* `next build --webpack` succeeds; `tsc --noEmit` clean.

## Phase 10 (Sprint 4 — Capacitor wrap) — TODO

* `capacitor.config.ts`, iOS/Android project init, splash + icon assets, native runtime testing on a device.

---

# 16. AI AGENT RULES

* DO NOT break local-first writes
* DO NOT expose service role key
* DO NOT use destructive sync
* DO NOT bypass RLS
* DO NOT store images in Postgres
* DO NOT mutate schema without migration
* DO NOT replace local repo without sync parity

---

END OF FILE
