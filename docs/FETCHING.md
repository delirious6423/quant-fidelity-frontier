# Fetch policy (local-first ledger)

Hugging Face rate-limits anonymous API use per IP. This project fetches each source **once** and works from local copies after that.

- **Ledger:** `cache/manifest.json` records every fetched object: `key` (repo@sha/path), source URL, fetch time, bytes, SHA-256, local path. Pinned by commit SHA, so a cached file never goes stale; a newer commit is a new key.
- **Local first:** every script reads `snapshots/` and `cache/`; nothing in the build or verification touches the network.
- **Fetch only what is missing:** the fetcher is given the list of keys already in the ledger and skips them.
- **Concurrency:** the direct CLI defaults to eight workers (`--workers 1` through `32`). Connector fetch tasks follow their own batch policy. The catalogue is fetched as a paginated listing, not one catalogue request per repository.
- **Rate limits:** the first HTTP 429 stops the batch. `cache/rate_limit.json` records the raw `Retry-After` and normalized delay. Already in-flight requests may finish; queued workers send no further requests. Completed ledger records survive locally and in the fetch workflow's checkpoint commit. Resume in a later run after the stated delay; 429 is never retried in a loop.
- **Transient failures:** HTTP 5xx and network retries are bounded to six attempts, with waits capped at 60 seconds. A failed fetch is not an empty source card. Only a confirmed HTTP 404 represents an absent README.
- **Credentials:** authorization stays on its original HTTP origin. Public CDN redirects lose credentials, HTTPS downgrades are rejected, and pagination cannot switch origin.
- **Refresh:** re-check `sha` via the single listing call; only repos whose `sha` changed are re-fetched.
