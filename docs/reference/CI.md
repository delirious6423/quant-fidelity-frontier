# CI: the verify-claims workflow

`.github/workflows/verify.yml` is the one CI workflow. It runs on every pull request and every push to `main`, and
`.github/workflows/pages.yml` calls it (`workflow_call`, `publish_pages_artifact: true`) and deploys the exact
artifact it tested. `tests/test_workflow_security.py` parses the workflow files and enforces the invariants below.

## Job graph

```
plan ──> model (one job per live model, matrix) ──┐
                                                  ├──> site ──> [Pages artifact, when called by pages.yml]
safety ───────────────────────────────────────────┘
```

| Job | Needs | What it runs |
|---|---|---|
| `plan` | – | `python3 pipeline/qcl_live.py > live-models.txt`, turned into the matrix JSON (`{"include": [{"dir", "slug"}, ...]}`). Models still `ingesting` are never built. A failing enumeration fails the job. |
| `safety` | – | The unit and workflow-safety suites that need no built site and no browser (assembly safety, workflow security, fetch security, gate integrity, source pins, mapping provenance, QA release, runner, dispatch, portable). |
| `model` | `plan` | For its one model: `QCL_ROOT=<dir> python3 web/build.py` (refuses on any gate failure), then `QCL_SITE=<dir>/site python3 tests/test_ui.py`. On failure the matching log lines go out as `::error` annotations titled with the model directory. The tested `<dir>/site` is uploaded as the artifact `site-<slug>` (1-day retention; `fail-fast: false`, so every model is reported). |
| `site` | `plan`, `safety`, `model` | Downloads every `site-*` artifact back into `models/<slug>/site` and fails if any live model has none (or an artifact belongs to a model that is not live), then runs every suite that needs built sites or `_site`, in the same order as before: benchmarks and disclosures UI, gates regression, claims, panel and column suites, `web/assemble.py _site`, benchmark view, README check, homepage, model page, discovery, comparison, the MCP suites and the MCP 1.x venv check. Last step: `actions/upload-pages-artifact` of `_site`, only when `publish_pages_artifact` is true. |

Guarantees kept from the single-job version, each asserted by `tests/test_workflow_security.py`:

- every `tests/test_*.py` suite runs on every PR and push to `main`, in exactly one job;
- the Pages artifact is uploaded by the `site` job alone, after its last check, and `site` needs every other job, so nothing is uploaded unless every check passed; the model jobs never see `publish_pages_artifact`;
- the `site` job never runs `web/build.py`: the bytes it assembles are the bytes `test_ui.py` passed on;
- no `${{ }}` expression is interpolated into a shell script (matrix values travel through the `MODEL_DIR` env variable);
- the checkout has no `ref:`, so a caller's immutable SHA is what gets built.

## Caching

Every job that installs Python packages restores `~/.cache/pip` with `actions/cache@v4`, keyed on the workflow file's
hash (with a `pip-<os>-` fallback), so `pip install` reuses the downloaded wheels. The `model` and `site` jobs then
read the installed Playwright version and restore `~/.cache/ms-playwright` under `playwright-<os>-<version>`: on a
hit only `playwright install-deps chromium` (system packages) runs; on a miss `playwright install --with-deps chromium`
downloads the browser as before. Caches are shared between branches through the default branch's cache.

## Measured wall time

Before (last green single-job run): 7 min 25 s total. Setup, pip and Playwright 31 s; "Assembly and workflow safety
regressions" 55 s; "Gates + build + UI tests, every live model" 171 s (9 models one after another; locally each build
takes 0.3 to 4.3 s and each `tests/test_ui.py` run 14 to 19 s); "Source-backed benchmarks and responsive
interactions" 40 s; gates regression 20 s; benchmark data and chart interactions 21 s; homepage 26 s; model page
layout 26 s; the MCP steps about 30 s together; everything else under 10 s each.

| Layout | Total wall time | Critical path | Setup per job |
|---|---|---|---|
| Single job (before) | 7 min 25 s | one job: 7 min 25 s | 31 s |
| Parallel jobs (after) | _to be filled from the first run_ | `plan` -> slowest `model` -> `site` | _cache hit / miss_ |

Expected after: `plan` about 20 s, the slowest `model` job about 70 s (setup, a 4 s build, a 19 s UI run and the
upload), and `site` about 3 min 45 s (setup, artifact download and the 188 s of suites that followed the builds), so
roughly 5 min 15 s on a cache miss and about 4 min 30 s on a hit. The `site` job is now the critical path; splitting
its suites further is the next lever.
