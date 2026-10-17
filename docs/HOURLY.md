# Hourly scan runbook

The scheduled task follows this file. It keeps every `live` model in `watchlist.json` current and ingests at most one
`queued` model per `min_hours_between_ingests`. It is cheap by design: the session itself runs scripts and makes
connector calls; card mapping goes to **Haiku subagents at high effort** (never Sonnet), and only for new or changed cards.

Rules that override everything below: never type a number or card text yourself (scripts copy them); a gate failure
quarantines an entry, never a hand edit; commit only gate-passing builds; never publish the Hugging Face Space.
Read `CLAUDE.md` first.

## 0. Setup
Clone the repo (attach `delirious6423/quant-fidelity-frontier` with push access), `pip install pillow numpy`, and set
`git config user.name Claude` and `git config user.email noreply@anthropic.com`. Let `TS` = UTC time `YYYYMMDDHH`.

## 1. Choose the work
- Every model with `status: live`: a **delta** run.
- If `now - last_ingest_at >= min_hours_between_ingests` hours, also the first `queued` model: an **ingest** run.
  Create `models/<slug>/model.json` first (model id, name, slug, a one-line `size_note`, the connector `pin_note` copied from
  `models/qwen3.8-27b/model.json`, and `others` links to every live model; add the new model to their `others` too).

## 2. Catalogue and delta (per model M = `models/<slug>`)
1. Call hf_fs once with `search hf://models --tag base_model:quantized:<MODEL> --limit 1000` plus six
   `cat hf://README.md --max-bytes 8737` padding ops, so the result is saved to a file. If it says truncated, also search
   with a query word per format (gguf, exl3, nvfp4, mlx, awq, gptq, fp8, int4, mxfp4), each padded the same way.
2. `python3 pipeline/qcl_catalogue.py M <result files>`: writes `M/cache/delta.json` (new or updated repos, gated ones
   excluded). If the delta is empty, skip to the next model.

## 3. Fetch the delta
Split `M/cache/delta.json` into shards of at most 100 ids (`M/cache/ids_<TS>a.json`, ...). For each shard, spawn one Haiku
agent (high effort) with: "Follow docs/agents/FETCH.md with S = <TS>a and M = models/<slug>." At most 4 at once.
Then `python3 pipeline/qcl_newmodel.py M` and `QCL_ROOT=M python3 pipeline/autoextract.py triage`.

## 4. Map new and changed cards
1. `QCL_ROOT=M python3 pipeline/detect_copies.py`, then `QCL_ROOT=M python3 pipeline/qcl_mapset.py`.
2. `QCL_ROOT=M python3 pipeline/qcl_queue.py work/<slug>` (gitignored work folder).
3. Spawn Haiku workers (high effort), at most 8 at once: "Follow docs/agents/MAP.md with W = work/<slug>, RUN = A, M = models/<slug>"
   (and the same with RUN = B). Each worker drains queues for its run. Relaunch until `work/<slug>/pending` is empty.
4. Run `python3 pipeline/qcl_mapcheck.py <queue> M/maps/<RUN>` for every claimed queue; re-queue any that fail.
5. `QCL_ROOT=M python3 pipeline/qcl_mapset.py --record`, then `QCL_ROOT=M python3 pipeline/qcl_watch.py merge`
   (both runs must agree; failures are quarantined to `M/audit/review_queue.json`), then `QCL_ROOT=M python3 pipeline/gen_auto.py`.

## 5. Build, test, publish
1. `for m in models/*/; do QCL_ROOT=${m%/} python3 build.py; done` and `python3 assemble.py`.
2. `python3 tests/test_gates.py`, `python3 tests/test_claims.py`, `python3 tests/test_mcp.py`, and
   `QCL_SITE=<site> python3 tests/test_ui.py` for each model whose data changed. Stop on any failure and report it.
3. Update `watchlist.json` (an ingested model becomes `live`; set `last_ingest_at`), add one line to `PROGRESS.md`
   (date, per model: new cards, values added, quarantined), commit, push, then
   `gh api -X POST repos/delirious6423/quant-fidelity-frontier/actions/workflows/pages.yml/dispatches -f ref=main`.
4. Final message: per model, cards fetched, values added or changed, entries quarantined; the deploy result.
