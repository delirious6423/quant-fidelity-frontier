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
