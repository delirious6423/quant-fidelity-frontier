# Fetch policy (local-first ledger)

Hugging Face rate-limits anonymous API use per IP. This project fetches each source **once** and works from local copies after that.

- **Ledger:** `cache/manifest.json` records every fetched object: `key` (repo@sha/path), source URL, fetch time, bytes, SHA-256, local path. Pinned by commit SHA, so a cached file never goes stale; a newer commit is a new key.
- **Local first:** every script reads `snapshots/` and `cache/`; nothing in the build or verification touches the network.
- **Fetch only what is missing:** the fetcher is given the list of keys already in the ledger and skips them.
- **Throttle:** one request at a time, ~0.4 s apart; the catalogue listing is one call (`/api/models?filter=base_model:quantized:...&expand[]=sha...`), not one call per repo.
- **Back off:** on HTTP 429 stop the whole batch, record `Retry-After`, and resume after it. Never retry in a loop.
- **Refresh:** re-check `sha` via the single listing call; only repos whose `sha` changed are re-fetched.
