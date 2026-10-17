# Developing

## Layout
| Path | What it is |
|---|---|
| `model.json`, `snapshots/`, `verified/`, `sets.py`, `audit/` | Data root for the first model (Flash-Next) |
| `models/<slug>/` | One data root per further model: `model.json`, `cache/cards/` (verbatim cards, SHA-256 pinned), `snapshots/index.json`, `verified/`, `audit/` |
| `verified/part*.py` | Claims, one `repo(...)` entry per repository; `part8_machine.py` is machine-mapped, `part9_auto.py` lists repos not yet reviewed |
| `build.py` | Gates + build for one data root (`QCL_ROOT=models/<slug>`) into its `site/` |
| `assemble.py` | Combines every model into one static site (`_site/`) with `models.json` and one sitemap |
| `qff_mcp/` | Read-only MCP server for agents |
| `qcl_*.py`, `autoextract.py` | Ingestion: fetch, extract candidates, map, merge, plan connector batches |
| `FASTPATH.md`, `NEW_MODEL_PROMPT.md`, `EXTRACTION_SPEC.md` | How a new model is ingested, and the extraction rules |

## Build and test
```bash
python3 build.py                                   # Flash-Next
QCL_ROOT=models/qwen3.8-27b python3 build.py        # 27B
python3 assemble.py                                # _site/: the whole static site
python3 test_gates.py && python3 test_claims.py && python3 test_ui.py && python3 test_mcp.py
```
Needs Python 3.11+, `pillow`, `numpy`; tests also need `playwright` and `mcp`.

## Hosting
GitHub Pages serves only static files, and that is all the site is (HTML, CSS, JS, JSON; data inlined in each page).
Python runs at build time only. `.github/workflows/pages.yml` builds in Actions and deploys `_site/`; it is
manual-trigger only (Settings → Pages → Source: GitHub Actions).
