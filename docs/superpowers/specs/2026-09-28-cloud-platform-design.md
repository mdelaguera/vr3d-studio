# Any2VR Cloud — Design (sub-project 4)

Status: approved in conversation 2026-09-28 (owner: mdelaguera). Parent spec:
`2026-09-27-platform-design.md`.

## Decisions

| Topic | Decision |
|---|---|
| GPU hosting | RunPod Serverless at launch; Vast.ai pool via SkyPilot in phase 2. Same worker container. |
| Payment model | Credit packs **and** a monthly subscription, one credit balance. |
| Job pricing | Fixed quote up front from a rate card; automatic refund on failure; real cost logged per job. |
| Client surfaces | Desktop app is the main client; small web portal for account, billing, history, downloads (incl. headset browsers). |
| Stack | Supabase (auth + Postgres), Cloudflare R2 (files), FastAPI on Fly.io (API), Next.js on Vercel (portal), Stripe, RunPod, n8n. |
| Credits system | Own append-only ledger in Postgres (no Autumn at launch). |

## Architecture

```
Desktop app ──┐                         ┌── RunPod Serverless (launch)
              ├─► API (FastAPI, Fly.io) ─► job queue ──┤   same worker container
Web portal ───┘                                        └── Vast.ai pool via SkyPilot (phase 2)
       API uses: Supabase (auth, Postgres) · Cloudflare R2 (pre-signed URLs) · Stripe · n8n
```

- The worker container runs the existing `engine` package: output identical to desktop/CLI.
- Media never passes through the API. Clients upload/download directly to/from R2 with
  pre-signed URLs; the API handles small JSON only.

## Job flow

1. **Quote:** client probes the source locally (duration, resolution) and sends it with model +
   format; API returns the price from the rate card and a short-lived quote id.
2. **Create:** user confirms; in one DB transaction the API checks balance, writes a `hold`
   ledger entry, and creates the job (`awaiting_upload`). Returns a pre-signed upload URL.
3. **Upload complete:** client notifies the API; the job is queued on RunPod (`queued`).
4. **Run:** worker downloads via pre-signed URL, converts, uploads the result, reports progress
   and final metrics (`running` → `succeeded` / `failed`).
5. **Settle:** success → `capture` ledger entry, result URL available for 24 h.
   Failure/timeout → `release` entry (full refund) and a user-readable reason.
6. Every job records GPU seconds, GPU type, cold-start seconds, and computed cost.

Job states: `awaiting_upload → queued → running → succeeded | failed | expired`.

## Billing

1 credit = $0.01.

| Product | Price | Credits |
|---|---|---|
| Starter pack | $10 | 1,000 |
| Plus pack | $25 | 2,750 (+10%) |
| Pro pack | $60 | 7,200 (+20%) |
| Subscription | $9 / month | 1,100 / month; unused roll over, capped at one month's grant |
| Signup gift | free | 50 (requires verified email) |

Launch rate card (stored in a `rates` table, editable without deploy):

| Job | Credits |
|---|---|
| Video, fast model, source ≤1080p | 10 / minute |
| Video, fast model, >1080p source | 25 / minute |
| Video, quality model (Marigold LCM / Hybrid) | 60 / minute |
| Photo, fast / quality | 1 / 3 each |

Minutes round up per started 10 seconds. Estimated fast-model cost ≈ $0.04/min on a serverless
RTX 4090 → ~55–65 % gross margin after Stripe fees. **These throughput assumptions must be
measured on the real worker (build step 2) and rates set from that data before launch.**

### Ledger

Append-only `ledger` table: `(id, user_id, kind, amount, job_id, stripe_event_id, created_at)`,
kinds `grant | purchase | subscription | hold | capture | release | adjustment`.
Balance = sum of `amount`; a hold is negative, `release` re-credits it, `capture` is a zero-amount
marker that finalizes the hold. Balance check + hold insert run in one serializable transaction
(no overspend under concurrency). Stripe webhooks are idempotent via a unique `stripe_event_id`.

## Models

- Conversion models: all Apache-2.0 (see parent spec catalog).
- Creation (sub-project 3): open models (commercial-OK licenses only) listed alongside frontier
  APIs, each labeled with where it runs and its license, priced as provider cost × margin.
  Never silently swap the provider the user chose.

## Privacy and security

- Inputs deleted when the job finishes; results 24 h after delivery (R2 lifecycle rules + a daily
  sweep). Content is never used for training or shared. Logs hold metadata only (sizes,
  durations, costs), never file names or frames.
- Supabase row-level security: users read only their own rows.
- Workers receive per-job pre-signed URLs, never storage credentials.
- Stripe webhook signatures verified; Stripe handles all card data.
- Rate limits on quote/job endpoints; concurrent jobs per user: 2 (packs), 4 (subscription).

## Failure handling

| Situation | Handling |
|---|---|
| Worker crash | Retry once on a fresh worker, then fail + refund |
| Upload never completes | Job `expired`, hold released after 2 h |
| No GPU capacity | Job stays `queued`; client shows queue position |
| Missed Stripe webhook | Nightly Stripe ↔ ledger reconciliation (n8n) |

## n8n automations

1. Welcome email + first-conversion tips.
2. Low-balance email (< 50 credits, at most weekly).
3. Failed-job alert to owner, grouped by error type.
4. Nightly: retention sweep verification + Stripe ↔ ledger reconciliation.
5. Weekly margin report per model and resolution (revenue vs GPU cost).
6. Phase 2: idle Vast.ai instance reaper.

## Build order

1. GPU stereo + projection in the engine (keeps a cloud GPU busy; also speeds up desktop).
2. Worker container + RunPod endpoint; benchmark real throughput; set rates.
3. Database schema, ledger, Stripe (test-first).
4. API: quote, jobs, status, download.
5. Desktop app: sign-in, "Convert in cloud", balance.
6. Web portal: sign-in, buy/subscribe, history, headset-friendly downloads.
7. n8n automations.
8. Phase 2: Vast.ai pool via SkyPilot.

## Testing

- Unit: rate card and quote math, ledger (including two jobs racing for the last credits),
  job state transitions.
- Stripe: test-mode webhooks against the handler, including duplicate delivery.
- Worker: end-to-end on RunPod with a short real clip; asserts output format and metrics.
- API: contract tests for every endpoint and auth/RLS denial cases.

## Accounts needed

Stripe, Cloudflare (R2), RunPod, Fly.io; existing Supabase and Vercel. Launch fixed cost target
under ~$30/month.
