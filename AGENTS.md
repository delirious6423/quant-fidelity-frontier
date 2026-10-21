# QFF: working notes for every agent

This file is the entry point for any agent or person working in this repository. It is not tied to one vendor or tool. `CLAUDE.md` only points here.

## What QFF is

QFF lines up every Hugging Face quant of a model by its publisher's own claims (KLD, perplexity, top-1, benchmarks, size) and shows which quants are efficient. Nothing is re-measured: each number is copied verbatim from the model card, with a link to the line it came from.

## Repository map

| Path | What it holds | Start reading |
|---|---|---|
| `/` (root) | `README.md`, this file, `CLAUDE.md` (points here), `CONTRIBUTING.md`, `LICENSE`, `PROGRESS.md` (status log; scans append to it), `watchlist.json`, `discovery.json`, `site.json` (configuration), `scan.lock` / `ingest.lock` / `qa.lock` (runtime locks committed by the scheduled tasks) | `README.md`, then `PROGRESS.md` |
| `web/` | Site generators: `build.py` (gates + one model's site), `assemble.py` (whole static site into `_site/`), `homepage.py`, `comparison.py`, `benchmarks.py`, `bench_tasks.py`, `benchmark_view.py`, `protocol_metadata.py`, `size_types.py`, `verified_protocol_sets.py`, `web_payload.py`, `discovery.py` | `web/README.md`, then `web/build.py` |
| `web/assets/` | CSS and JS the generators read (inlined into the pages, except `home.css` and `home.js`, copied beside the homepage): `site_style.css`, `site_script.js`, `site_explorer.js`, `site_benchmarks.css`, `site_benchmarks.js`, `benchmark_view.css`, `benchmark_view.js`, `home.css`, `home.js` | `site_script.js` |
| `gates/` | Fail-closed checks: `verify.py`, `verify_columns.py`, `verify_images.py`, `current.py`, `gatecache.py` | `gates/verify.py` |
| `pipeline/` | Ingestion and scan scripts: collection, mapping, merge and publish, scheduled-task state machines, scripted runner | `pipeline/README.md` (index by stage) |
| `tests/` | Every regression suite. `.github/workflows/verify.yml` lists which ones CI runs and in what order | `.github/workflows/verify.yml` |
| `qff_mcp/` | The read-only MCP server and its Hugging Face Space scaffold (`space/`) | `qff_mcp/README.md` |
| `docs/runbooks/` | How to run things: `HOURLY.md` (the three scheduled tasks), `PORTABLE_SCAN.md` (the agent-agnostic loop), `SCRIPTED_RUNNER.md`, `FASTPATH.md`, `NEW_MODEL_PROMPT.md`, `FETCHING.md` | `HOURLY.md` |
| `docs/runbooks/agents/` | Worker instructions: `FETCH.md`, `MAP.md`, `REVIEW.md`, `README_CHECK.md` | The file named in the prompt you were given |
| `docs/reference/` | What things mean: `METHOD.md`, `DEVELOPING.md`, `EXTRACTION_SPEC.md`, `COMPARISON_CLASSIFICATIONS.md`, `COMPARISON_VALIDATION.md`, `BENCHMARKS.md`, `BENCHMARK_COVERAGE.md`, `CROSS_MODEL_BENCHMARKS.md`, `DISCOVERY.md`, `SPONSORS.md` | `METHOD.md`, then `DEVELOPING.md` |
| `docs/history/` | Past reviews and results: `PLAN.md`, `HOLISTIC_REVIEW.md`, `GLM_BENCHMARK_PILOT.md`, `review/`, `scripted-runner/`, `previews/`, `legacy-tools/` | Only when a task refers to it |
| `models/<slug>/` | One data root per model: `model.json`, `cache/cards/` or `snapshots/` (cards, SHA-256 pinned), `verified/` (claims), `maps/`, `audit/`, `site/` (build output, git-ignored) | `model.json` |
| `logs/scan/` | Scheduled-task command logs | The newest file |
| `audit/` | Deployment receipts (`deployed.json`, `pending_release.json`) | `pipeline/deploy_state.py` |
| `protocol_evidence/` | Protocol source bindings, one JSON file per model | `docs/reference/COMPARISON_CLASSIFICATIONS.md` |

## Working rules

- Claims only: never type a number into `verified/`. Values are copied by code (`pipeline/qcl_map.py apply`, `pipeline/qcl_watch.py merge`) from card text. A gate failure means the entry is quarantined, never "fixed" by hand-editing a number.
- Never rebuild a card from memory or a truncated tool result. Cards are saved byte-for-byte by `pipeline/qcl_mcpsave.py` and pinned by SHA-256.
- Data roots: every model lives in `models/<slug>/`. Flash-Next is the default when `QCL_ROOT` is unset. Code lives in `gates/`, `pipeline/`, `web/` and `tests/`; the layout is in `docs/reference/DEVELOPING.md`.
- Before committing, run the build and the tests:
  ```bash
  for d in $(python3 pipeline/qcl_live.py); do QCL_ROOT=$d python3 web/build.py; done   # every live model (models still ingesting are never built)
  python3 web/assemble.py _site
  python3 pipeline/readme_sync.py --check
  ```
  then the test suites with the commands exactly as listed in `.github/workflows/verify.yml`. At minimum: `tests/test_gates.py`, `tests/test_claims.py`, `tests/test_ui.py` (each site via `QCL_SITE=models/<slug>/site`), `tests/test_mcp.py` (with `QFF_DATA=_site`), `tests/test_discovery.py`.
- The repo is public: commit no secrets, tokens, personal data, local paths or scratch files.
- Pages deploys (`deploy-pages` workflow) are approved by the owner. Publishing the Hugging Face MCP Space needs the owner's explicit go-ahead in chat.
- Agent fan-out: at most 20 concurrent workers. Mapping workers claim queues only with `pipeline/qcl_claim.py` (atomic) and write only their claimed names.
- Status and next steps live in `PROGRESS.md`.

## Running the pipeline

- The three scheduled tasks (live updates, new model, QA and publish): `docs/runbooks/HOURLY.md`. The entry points are `pipeline/qcl_next.py` and `pipeline/qcl_qa.py`; each prints what to do next, and the session does only that.
- The agent-agnostic loop (no connector, no subagent tool, no credentials, nothing published): `docs/runbooks/PORTABLE_SCAN.md`. The entry point is `pipeline/qcl_portable.py`.

## Where new files go

- Site generators: `web/`. Their CSS and JS: `web/assets/`.
- Ingestion, scan and publish scripts: `pipeline/`. Add the script to the matching table in `pipeline/README.md`.
- Fail-closed checks: `gates/`.
- Tests: `tests/`, and every new `tests/test_*.py` is registered in `.github/workflows/verify.yml` (a coverage check rejects omitted suites).
- Documentation under `docs/`: a runbook (how to run something) goes in `docs/runbooks/`, worker instructions in `docs/runbooks/agents/`; a reference (what something means) goes in `docs/reference/`; a past review or result goes in `docs/history/`.
- Model data: `models/<slug>/`. Values there are copied by the pipeline scripts, never typed (see the working rules).
