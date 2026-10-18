# Developing

## Layout
| Path | What it is |
|---|---|
| `build.py` | Gates + build for one model (`QCL_ROOT=models/<slug>`, default Flash-Next) into `models/<slug>/site/` |
| `assemble.py` | Combines every model into one static site (`_site/`): landing page, one folder per model, `models.json`, one sitemap |
| `models/<slug>/` | One data root per model: `model.json` (name, slug, notes, links), card snapshots (`snapshots/` or `cache/cards/`, SHA-256 pinned), `verified/` (claims), `sets.py`, `audit/` |
| `gates/` | Fail-closed checks: `verify.py`, `verify_columns.py`, `verify_images.py`, and `gatecache.py` (parallel runs, content-hash cache) |
| `pipeline/` | Ingestion: `qcl_fetch.py`, `qcl_stats.py` + `qcl_plan.py` + `qcl_mcpsave.py` (connector fetch), `qcl_newmodel.py`, `autoextract.py`, `qcl_map.py`, `qcl_watch.py` (merge), `detect_copies.py`, `gen_auto.py` |
| `web/` | Site template: `site_style.css`, `site_script.js`, `site_explorer.js` |
| `tests/` | `test_gates.py`, `test_claims.py`, `test_ui.py`, `test_mcp.py` |
| `qff_mcp/` | Read-only MCP server for agents, plus a Hugging Face Space scaffold |
| `legacy/` | One-off tools from the first (hand-verified) Flash-Next build, kept for the audit trail |
| `docs/` | Method, this file, the fast path (`FASTPATH.md`), fetch policy, extraction spec, new-model prompt |

In `verified/`, `part8_machine.py` is machine-mapped and `part9_auto.py` lists repos not yet reviewed; both are generated, never hand-edited.

## Build and test
```bash
for d in $(python3 pipeline/qcl_live.py); do QCL_ROOT=$d python3 build.py; done   # every live model: gates + site
python3 assemble.py                                            # _site/: the whole static site
python3 tests/test_gates.py && python3 tests/test_claims.py && python3 tests/test_mcp.py
for d in $(python3 pipeline/qcl_live.py); do QCL_SITE=$d/site python3 tests/test_ui.py; done
```
Needs Python 3.11+, `pillow`, `numpy`; tests also need `playwright` and `mcp`.

## Adding a model
`python3 pipeline/qcl_newmodel.py models/<slug>` turns a fetched card cache into a data root; then follow
`docs/FASTPATH.md` (extract, map twice, merge, build). Add the model to each other model's `others` in `model.json`.

## Hosting
GitHub Pages serves only static files, and that is all the site is (HTML, CSS, JS, JSON; data inlined in each page).
Python runs at build time only. `.github/workflows/pages.yml` builds in Actions and deploys `_site/` (manual trigger;
Settings → Pages → Source: GitHub Actions).

## Site analytics
Optional and cookie-free (Umami Cloud). Set `analytics.website_id` in `site.json` to turn it on; the script is added
only to the public pages (never the claude.ai previews) and counts only on `delirious6423.github.io`, so local and CI
test runs are not recorded. Outbound GitHub and support clicks and explorer set changes are recorded as events.
