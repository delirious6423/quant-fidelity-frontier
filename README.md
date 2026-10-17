# QFF: Quant Fidelity Frontier

What publishers claim about their quants, pinned to the line they said it.

Hugging Face hosts hundreds of quantizations of every popular model, and each publisher reports quality its own way:
KLD against BF16, top-1 agreement against someone's Q8_0, wikitext perplexity, a few benchmarks, or nothing. QFF lines
those claims up per base model, shows which quants are efficient (nothing smaller is better), and says plainly when two
numbers cannot be compared.

QFF does **not** re-measure anything. Every number is a publisher's own claim, copied verbatim by code from the model
card, with the card line it came from, the reference model, the eval data and any problems with the method.

| Model | Repos | Claimed values | Page |
|---|---|---|---|
| Qwen/Qwen3.8-Flash-Next | 453 | 2,980 | `/` |
| Qwen/Qwen3.8-27B | 1,220 | 4,747 | `/qwen3.8-27b/` |

Site (when published): https://delirious6423.github.io/quant-fidelity-frontier/

## How to read the site
- **Overview (default):** every claim on one chart. Measurers, references, datasets and methods differ, so it is for a
  first glance only and is labelled *not comparable*.
- **Measurement sets:** one measurer, one reference, one dataset, one method. Only inside a set can quants be ranked;
  the frontier and "Beaten by" columns appear only there. Sets are sorted by owner.
- **Ledger:** every row with its source link. **Publisher notes:** method, flags, copies and cards with no claims.

## Rules for a claim
1. It must be in the publisher's own card (or a cited docs page) at the captured version.
2. The string is kept exactly as printed (`0.0771 ± 0.0007`, `86.95 ± 0.14 %`), never rounded or retyped.
3. It cites its line, and records the reference model, eval data, context and tool as stated, or `not stated`.
4. Numbers a publisher reports for someone else's file are `cross` rows; copies of other cards are labelled as copies.
5. Flags describe what the card says, with a line reference. No guesses about motives.

## Fail-closed gates
`build.py` refuses to build if any gate fails:
- `verify.py`: each value appears verbatim on its cited line, with the right sign and a unit next to every size.
- `verify_columns.py`: table values sit under a header of the right metric, in the cited row.
- `verify_images.py`: values read from chart images match marker positions (only for models with image claims).
- Plausibility and sign assertions inside `build.py`.

Tests: `test_gates.py` (gate mutations), `test_claims.py` (review regressions), `test_ui.py` (Playwright, desktop and
phone), `test_mcp.py` (MCP server answers checked against the data). CI runs all of them (`.github/workflows/verify.yml`).

## Layout
| Path | What it is |
|---|---|
| `model.json`, `snapshots/`, `verified/`, `sets.py`, `audit/` | Data root for the first model (Flash-Next) |
| `models/<slug>/` | One data root per further model: `model.json`, `cache/cards/` (verbatim cards, SHA-256 pinned), `snapshots/index.json`, `verified/`, `audit/` |
| `verified/part*.py` | Claims: one `repo(...)` entry per repository; `part8_machine.py` is machine-mapped, `part9_auto.py` lists repos not yet reviewed |
| `build.py` | Gates + build for one data root (`QCL_ROOT=models/<slug>`) into its `site/` |
| `assemble.py` | Combines every model into one static site (`_site/`) with `models.json` and one sitemap |
| `qff_mcp/` | Read-only MCP server for agents (see below) |
| `qcl_*.py`, `autoextract.py` | Ingestion: fetch, extract candidates, map, merge, plan connector batches |
| `FASTPATH.md`, `NEW_MODEL_PROMPT.md`, `EXTRACTION_SPEC.md` | How a new model is ingested, and the extraction rules |

## Build locally
```bash
python3 build.py                                   # Flash-Next: gates + site/
QCL_ROOT=models/qwen3.8-27b python3 build.py        # 27B: gates + models/qwen3.8-27b/site/
python3 assemble.py                                # _site/: the whole static site
python3 test_gates.py && python3 test_claims.py && python3 test_ui.py && python3 test_mcp.py
```
Needs Python 3.11+, `pillow`, `numpy`; tests also need `playwright` and `mcp`.

## Hosting
GitHub Pages serves only static files, and that is all the site is: HTML, CSS, JavaScript and JSON (data is inlined in
each page). Python runs at build time only. `.github/workflows/pages.yml` builds in Actions and deploys `_site/`; it is
manual-trigger only, so nothing goes public until the owner runs it with Settings → Pages → Source: GitHub Actions.

## For agents: MCP server
`qff_mcp/qff_mcp.py` exposes `list_models`, `search_quants`, `get_repo`, `list_sets`, `frontier` and `best_under`
over stdio or streamable HTTP. It reads the published `models.json` and `data.json`, so it needs no database.
```bash
claude mcp add qff -e QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/ -- python3 qff_mcp/qff_mcp.py
```
See `qff_mcp/README.md` for Claude Desktop config and the Hugging Face Space for a hosted connector URL.

## Corrections
Found a wrong value or a missing quant? Open an issue with the repo and the card line. Each claim lives in a plain data
file next to its source line, so a fix is a one-line change that the gates re-check.

Licence: code MIT; the compiled dataset CC BY 4.0. The claims themselves belong to their publishers.
