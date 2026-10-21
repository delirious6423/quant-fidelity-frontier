# Developing

## Layout
| Path | What it is |
|---|---|
| `web/` | Site generators: `build.py` (gates + build for one model, `QCL_ROOT=models/<slug>`, default Flash-Next, into `models/<slug>/site/`), `assemble.py` (every live model into one static site, `_site/`: landing page, one folder per model, `models.json`, one sitemap), and the modules they import (`homepage.py`, `comparison.py`, `benchmarks.py`, `bench_tasks.py`, `benchmark_view.py`, `protocol_metadata.py`, `size_types.py`, `verified_protocol_sets.py`, `web_payload.py`, `discovery.py`) |
| `web/assets/` | CSS and JS shipped with the site: `site_style.css`, `site_script.js`, `site_explorer.js`, `site_benchmarks.css`, `site_benchmarks.js`, `benchmark_view.css`, `benchmark_view.js`, `home.css`, `home.js` |
| `gates/` | Fail-closed checks: `verify.py`, `verify_columns.py`, `verify_images.py`, `current.py`, and `gatecache.py` (parallel runs, content-hash cache) |
| `pipeline/` | Ingestion and scan scripts, indexed by stage in `pipeline/README.md`: collection (`qcl_fetch.py`, `qcl_catalogue.py`, `qcl_stats.py`, `qcl_plan.py`, `qcl_mcpsave.py`, `qcl_newmodel.py`), mapping (`autoextract.py`, `qcl_map.py`, `qcl_queue.py`, `qcl_claim.py`, `detect_copies.py`, `gen_auto.py`), merge and publish (`qcl_watch.py`, `qcl_live.py`, `readme_sync.py`), the scheduled-task state machines (`qcl_next.py`, `qcl_scan.py`, `qcl_qa.py`) and the scripted runner (`qcl_runner.py`, `qcl_portable.py`, `runner_*.py`) |
| `tests/` | Every regression suite; `.github/workflows/verify.yml` lists which ones CI runs and in what order |
| `qff_mcp/` | Read-only MCP server for agents, plus a Hugging Face Space scaffold (`space/`) |
| `docs/runbooks/` | How to run things: `HOURLY.md` (the three scheduled tasks), `PORTABLE_SCAN.md` (the agent-agnostic loop), `SCRIPTED_RUNNER.md`, `FASTPATH.md`, `NEW_MODEL_PROMPT.md`, `FETCHING.md`; worker instructions in `agents/` (`FETCH.md`, `MAP.md`, `REVIEW.md`, `README_CHECK.md`) |
| `docs/reference/` | What things mean: `METHOD.md`, this file, `EXTRACTION_SPEC.md`, `COMPARISON_CLASSIFICATIONS.md`, `COMPARISON_VALIDATION.md`, `BENCHMARKS.md`, `BENCHMARK_COVERAGE.md`, `CROSS_MODEL_BENCHMARKS.md`, `DISCOVERY.md`, `SPONSORS.md` |
| `docs/history/` | Past reviews and results: `PLAN.md`, `HOLISTIC_REVIEW.md`, `GLM_BENCHMARK_PILOT.md`, `review/`, `scripted-runner/`, `previews/`, and `legacy-tools/` (one-off tools from the first, hand-verified Flash-Next build, kept for the audit trail) |
| `models/<slug>/` | One data root per model: `model.json` (name, slug, notes, links), card snapshots (`snapshots/` or `cache/cards/`, SHA-256 pinned), `verified/` (claims), `maps/`, `sets.py`, `audit/`, `site/` (build output, git-ignored) |
| `logs/scan/` | Scheduled-task command logs |
| `audit/` | Deployment receipts (`deployed.json`, `pending_release.json`) |
| `protocol_evidence/` | Protocol source bindings, one JSON file per model |

In `verified/`, `part8_machine.py` is machine-mapped and `part9_auto.py` lists repos not yet reviewed; both are generated, never hand-edited.

## Build and test
```bash
for d in $(python3 pipeline/qcl_live.py); do QCL_ROOT=$d python3 web/build.py; done   # every live model: gates + site
python3 web/assemble.py _site                                                          # the whole static site
python3 pipeline/readme_sync.py --check                                                # README table, links
python3 tests/test_gates.py && python3 tests/test_claims.py && python3 tests/test_discovery.py
for d in $(python3 pipeline/qcl_live.py); do QCL_SITE=$d/site python3 tests/test_ui.py; done
QFF_DATA=_site python3 tests/test_mcp.py
```
`.github/workflows/verify.yml` lists every suite and the order CI runs them; the commands above are the minimum before a commit.
Needs Python 3.11+, `pillow`, `numpy`; tests also need `playwright` and `mcp`.

## Adding a model
`python3 pipeline/qcl_newmodel.py models/<slug>` turns a fetched card cache into a data root; then follow
`docs/runbooks/FASTPATH.md` (extract, map twice, merge, build). Add the model to each other model's `others` in `model.json`.

## Hosting
GitHub Pages serves only static files, and that is all the site is (HTML, CSS, JS, JSON; data inlined in each page).
Python runs at build time only. `.github/workflows/pages.yml` builds in Actions and deploys `_site/` (manual trigger;
Settings → Pages → Source: GitHub Actions).

Publishing invokes the reusable `verify.yml` workflow and deploys its exact tested
artifact. A failed test cannot be bypassed by rebuilding in a separate deployment
job. Keep every `tests/test_*.py` suite in that workflow; a coverage check rejects
omitted suites.
The workflow's job graph (parallel per-model builds, one final `site` job that uploads
the artifact), its caches and its measured times are in `docs/reference/CI.md`.

Assembly requires every live model to have valid build output. Its destination
must be new, empty, or an existing QFF assembly; source directories and unrelated
files are refused. It stages and validates the complete site before replacing the
previous output, and restores that output if the final replacement fails.

The browser receives a smaller chart projection from `web_payload.py`. Full point
protocols and classifier diagnostics remain in `data.json`; numerical claims,
source links and displayed per-metric evidence remain in both. Run
`python3 tests/benchmark_payload.py --site _site` for a repeatable V8 JSON parsing
benchmark. It measures payload parsing, not whole-page or network latency.

## Site analytics
Optional and cookie-free (Umami Cloud). Set `analytics.website_id` in `site.json` to turn it on; the script is added
only to the public pages (never the claude.ai previews) and counts only on `delirious6423.github.io`, so local and CI
test runs are not recorded. Outbound GitHub and support clicks and explorer set changes are recorded as events.

## Native quantization and source-resolved claims

Direct Hub ingestion now preserves commit revisions, exact card hashes, and paginated file trees.
Use repeatable `qcl_fetch.py --include-repo owner/repo` for source-checked derivatives missing from the base-model tag.
Failed HTTP fetches are not empty cards. `qcl_newmodel.py` accepts direct and connector caches.

A model config may set `max_size_gb`, `native_reference_patterns`, and `named_metric_series`.
DeepSeek V4.1 Flash opts into these: its native reference is already mixed precision, storage can exceed
400 GB, and corpus-specific fidelity metrics must retain their own series. Entry methods and panels
remain distinct; Engram-only, decoder-only and routed-expert sizes have separate bases.

After two mapping runs, an independent resolver may write `maps/resolved/<repo>.map.json`.
`maps/resolved/manifest.json` has a `repos` object keyed by repository, with `source_sha256` and
`mapping_sha256` plus reviewer provenance. The merge applies an overlay only when both hashes match,
keeps the original disagreements and resolution provenance in the audit, then runs normal quarantine gates.
Never rewrite A/B mappings to manufacture agreement. Supplemental candidates must be extracted from the
exact pinned source and retain its line/header/row binding.

## Comparison evidence

See [COMPARISON_CLASSIFICATIONS.md](COMPARISON_CLASSIFICATIONS.md) before adding protocol assertions. Never infer compatibility from a publisher or missing methodology. Review source quotes, hashes, row scope, metric aliases and contradictions. New evidence changes eligibility only; it must not modify published values. Run `test_comparison.py`, `test_protocol_metadata.py`, `test_verified_protocol_sets.py`, `test_comparison_ui.py` and `test_mcp_comparison.py` as well as the existing gates/UI/MCP suites. The shared classifier ships with the read-only MCP package; Pages remains static.
