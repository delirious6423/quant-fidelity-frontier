# Progress

## Done (2026-10-18)
- Qwen3.8-Flash-Next: 453 repos, 2,980 values, two external reviews folded in; all gates and tests pass.
- Qwen3.8-27B: 1,220 of 1,234 cards captured (14 gated, access denied). 599 unique cards with metrics mapped by two independent runs; 3,801 agreed values published, 142 entries quarantined by gates, 1,647 single-run values held (`models/qwen3.8-27b/audit/review_queue.json`). 621 repos shown as not yet reviewed.
- Site: overview of every claim is the default view; sets sorted by owner; VRAM-sized presets (8–256 GB); QFF rename.
- Static Pages pipeline (`assemble.py`, `pages.yml`, manual trigger) and MCP server (`qff_mcp/`, `test_mcp.py`).
- Live: https://delirious6423.github.io/quant-fidelity-frontier/ (landing page; each model at /<slug>/). First deploy 2026-10-18; CI green on GitHub runners.
- Repo reorganized: every model under `models/<slug>/`; code in `gates/`, `pipeline/`, `web/`, `tests/`, `legacy/`; docs in `docs/`. Built sites are no longer committed (CI and the Pages workflow build them).

## Hourly scan fixes (2026-10-18)
- Runs from 22:04 to 04:04 UTC all failed or hung. Causes found: (1) Flash-Next had no mapping record, so every run queued all 249 of its hand-reviewed cards for Haiku mapping (62 queues); (2) the first ingest (DeepSeek-V4.1-Flash, due from 00:30 UTC) was attempted in one run and discarded on failure, together with the live updates; (3) a dead session's lock blocked two more runs (2 h lock).
- Fixed: Flash-Next baseline in `maps/mapped.json` (`qcl_mapset.py --baseline`; a no-change prepare now queues 0 cards); runbook split into Part A (live updates, committed on their own) and Part B (ingest, time-boxed and resumable via `ingest-state`, `fetched`, `checkpoint`; status `ingesting`); `elapsed` time budget (stop new work at 40 min, workers at 45); lock taken over after 75 min; CI/Pages build live models only (`qcl_live.py`); `qcl_newmodel.py` keeps index fields like `base` and no longer marks a card current when its refetch failed; UI tests run offline (external fonts/analytics blocked); `publish` also runs `test_discovery.py`.

## DeepSeek V4.1 Flash (PR #2, merged 2026-10-18)
- 136 repositories catalogued; 133 pinned cards captured; three access-denied cards excluded.
- 61 distinct cards mapped independently twice; 42 resolver maps adjudicate differences against pinned sources. 72 cards remain explicitly unreviewed.
- 1,166 source-linked values, 130 own variants; zero held disagreements and zero quarantined entries after source/column gates.
- Added native-reference, Engram storage scope, per-corpus panel, and top-1 ranking support. No new measurements were run.
- Local builds, claim/gate regressions and MCP checks pass. Browser validation runs in PR CI because the local Chromium download was unavailable.
- Merge review (2026-10-18, independent audit: 66 sampled values read in context, all 1,219 checked by code, 0 absent from their cited line): fixed before merge: prose values right after a qualifier (~, ≈, <, about) are no longer published bare; approximate or ranged sizes are held (`size_held`) instead of shown as a point; resolver-written candidates keep their raw strings only after a whole-value check on the card line, and their typed header/label text is never published as card wording; strict per-corpus agreement is opt-in (`agree_strict_corpus` in model.json, on for DeepSeek) because it would hold ~500 agreed 27B values; scratch `PLAN.md`, a 0-byte card and a duplicate base card removed. Result: 1,121 published values (was 1,166), 5 entries quarantined. Regression test: `tests/test_qualifiers.py`.

## Scan check (2026-10-18 07:40 UTC)
- 07:03 run (first on the new runbook, 24 min): fetched all 174 GLM-5.3-Flash cards and mapped 87 twice (A/B), then checkpointed. It could not go live: `build.py` rejected GLM's 642 GB BF16 files against the 400 GB size ceiling, and two UI checks assumed Qwen-style data. No live-model update was committed in that run (cause not recorded).
- Fixed: `plan` writes `max_size_gb` into a new model.json (2.5 GB per billion parameters in the id, else 1000; GLM set to 1000); explorer keeps the legend when every quant is hidden (otherwise the hidden repo could not be shown again); UI checks read millinat cells in their unit and choose the search words from the model's data; every `qcl_scan.py` command now logs to `logs/scan/<UTC>.jsonl`, committed with the unlock, so the next check can see which steps ran.

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
