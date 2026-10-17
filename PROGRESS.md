# Progress

## Done (2026-10-18)
- Qwen3.8-Flash-Next: 453 repos, 2,980 values, two external reviews folded in; all gates and tests pass.
- Qwen3.8-27B: 1,220 of 1,234 cards captured (14 gated, access denied). 599 unique cards with metrics mapped by two independent runs; 3,801 agreed values published, 142 entries quarantined by gates, 1,647 single-run values held (`models/qwen3.8-27b/audit/review_queue.json`). 621 repos shown as not yet reviewed.
- Site: overview of every claim is the default view; sets sorted by owner; VRAM-sized presets (8–256 GB); QFF rename.
- Static Pages pipeline (`assemble.py`, `pages.yml`, manual trigger) and MCP server (`qff_mcp/`, `test_mcp.py`).
- Live: https://delirious6423.github.io/quant-fidelity-frontier/ (landing page; each model at /<slug>/). First deploy 2026-10-18; CI green on GitHub runners.
- Repo reorganized: every model under `models/<slug>/`; code in `gates/`, `pipeline/`, `web/`, `tests/`, `legacy/`; docs in `docs/`. Built sites are no longer committed (CI and the Pages workflow build them).

## Next
1. Hourly scan: scheduled task "QFF hourly scan" (cloud, minute 3 of every hour) follows `docs/HOURLY.md`. Watch list approved by the owner (`watchlist.json`); one queued model ingested per 5 hours, in list order; Haiku subagents only.
2. 27B held values: resolver pass (Haiku, high effort), core metrics first (~540 values), then benchmarks.
3. 27B wave-2 cards (benchmark-only, 251) and cold review of the 27B data.
4. Hugging Face MCP Space: only on the owner's go-ahead. (Repo is public; Pages deploy approved 2026-10-18.)

## Discovery implementation (2026-10-18)
- Branch `improve-discovery`: search-focused titles, visible comparability/VRAM caveats, three static reading guides, sitemap entries and generated agent entrypoint.
- `pipeline/export_hf_dataset.py` prepares a provenance-preserving dataset/card locally; no external publication.
- `tests/test_discovery.py` checks canonical routes and exact exported claim preservation; included in CI.
- Remaining account/deployment actions and exact metadata command: `docs/DISCOVERY.md`. No indexing, traffic growth or connector listing is claimed.
- Review finding: all 1,341 27B claim rows currently use mutable `main` source refs; the new export exposes `source_revision_pinned` and explains the limitation. No source records were changed.
- Validation: both model builds; gate and claim suites; MCP 15/15; discovery 7 canonical pages and 2,444 exact exported claims. Browser tests require stubbing external HTTPS resources in this environment because resource fetches return ERR_EMPTY_RESPONSE.
