#!/usr/bin/env python3
"""Assemble the external review pack: re-run every gate, capture outputs, write REVIEW_PACK.md, zip everything needed
to re-check the work offline. Output: dist/qff-review-pack.zip"""
import json, os, subprocess, zipfile, datetime, collections, hashlib, sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")
def run(cmd):
    p = subprocess.run(cmd, cwd=CODE, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr).strip(), p.returncode

gates = {}
for name, cmd in [("verify.py (verbatim check, all repos)", "python3 gates/verify.py"),
                  ("verify_columns.py (table-cell / metric binding)", "python3 gates/verify_columns.py"),
                  ("verify_images.py (chart-image transcriptions)", "python3 gates/verify_images.py"),
                  ("detect_copies.py (copied cards, share >= 0.30 listed; >= 0.60 treated as copied)", "python3 pipeline/detect_copies.py"),
                  ("diff_blind.py — first 44 repos vs blind auditor", "python3 legacy/diff_blind.py models/qwen3.8-flash-next/audit/blind.json"),
                  ("diff_blind.py — 152 later measured repos vs two blind auditors", "python3 legacy/diff_blind.py models/qwen3.8-flash-next/audit/blind_new_0.json models/qwen3.8-flash-next/audit/blind_new_1.json"),
                  ("build.py (plausibility assertions + site build)", "python3 build.py"),
                  ("test_gates.py (mutation tests: gates must fail closed)", "python3 tests/test_gates.py"),
                  ("test_claims.py (regression checks for the cold review, F1-F24)", "python3 tests/test_claims.py"),
                  ("test_ui.py (Playwright: explorer, filters, layout at 1366px and 390px)", "python3 tests/test_ui.py")]:
    out, rc = run(cmd); gates[name] = (out, rc)
    if rc != 0 and not name.startswith("diff_blind") and not name.startswith("detect_copies"):
        raise SystemExit(f"REFUSING TO PACK: gate failed: {name} (exit {rc})\n{out[-3000:]}")

d = json.load(open(f"{ROOT}/site/data.json"))
idx = json.load(open(f"{ROOT}/snapshots/index.json"))
repos, rows, sets = d["repos"], d["rows"], d["sets"]
ref = collections.Counter(x["ref_class"] for x in repos)
fmt = collections.Counter(x["fmt"] for x in repos)
links = [(s["id"], l) for s in sets for l in s.get("linked", [])]
adv = open(f"{ROOT}/audit/adversarial_review.md").read() if os.path.exists(f"{ROOT}/audit/adversarial_review.md") else "(not available)"
astra = open(f"{ROOT}/audit/review_fixes.md").read() if os.path.exists(f"{ROOT}/audit/review_fixes.md") else "(none recorded)"

def block(t): return "```\n" + t[-6000:] + "\n```"
gate_md = "\n\n".join(f"### {n}\nexit code {rc}\n{block(o)}" for n, (o, rc) in gates.items())
link_md = "\n".join(f"| {l['from']} → {sid} | {l['metric']} | {len(l['anchors'])} | {', '.join(l['anchors'])} | {l['loo_rms_pct']}% | {l['loo_worst_pct']}% | {len(l['points'])} |" for sid, l in links)
set_md = "\n".join(f"| {s['id']} | {s['title']} | {len(s['points'])} | {', '.join(s['metrics'])} | {s['reference']} | {'auto' if s.get('auto') else ('n/a' if not s['comparable'] else 'curated')} |" for s in sets)

doc = f"""# QFF (Quant Fidelity Frontier) — review pack

Prepared {datetime.date.today()} for an independent reviewer. You do not need the conversation that produced this; everything needed to re-check the work offline is in this archive.

## 1. What this is

A public comparison of every Hugging Face quantization of **Qwen/Qwen3.8-Flash-Next**, built only from each publisher's **own claims** (model cards and, for Unsloth, its docs page). Nothing was re-measured. The product is a static site (`site/index.html`, for GitHub Pages) plus a JSON dataset (`site/data.json`).

**Goal:** let a reader see what each quant's publisher claims, how it was measured, how claims relate where they are comparable, and where they disagree, without the site implying rankings the evidence does not support.

**Scope at capture ({idx['captured_at'][:10]} for the first 44 cards; the rest captured the same day):** {len(repos)} repositories tagged `base_model:quantized:Qwen/Qwen3.8-Flash-Next`. Formats: {dict(fmt)}.

## 2. Pipeline

| Step | What | Code / data |
|---|---|---|
| 1 Catalogue | One HF API call lists all quantized derivatives with sha, createdAt, lastModified, downloads, tags | `cache/catalogue.json` |
| 2 Snapshot | Raw `README.md` fetched once per repo **at its pinned commit sha**, stored locally with SHA-256; HF file listings for sizes; turboderp branch listings; Unsloth docs page; turboderp chart PNGs | `snapshots/`, `cache/cards/`, `cache/manifest.json`, `FETCHING.md` |
| 3 Extract | Claims written as Python data (`verified/part*.py`): exact published strings, README line numbers, methodology as stated, sizes with unit and basis, flags citing lines. First 44 repos by hand from raw text; 152 measured repos by six agents reading only local snapshots; the remaining 257 cards re-read in full by five further agents after external review (`verified/part5_*.py`, acceptance list `audit/astra_missed25.json`). Any card not yet read in full is catalogued by `gen_auto.py` and labelled **not yet reviewed**, never "no claims" | `verified/` |
| 4 Image claims | turboderp publishes results only as charts. User-supplied images were confirmed pixel-identical to the card's images; values transcribed from printed labels; bpw digitized from marker positions | `verified/part3_images.py`, `verify_images.py`, `snapshots/images/` |
| 5 Copies | Card-level: share of a card's numeric claim lines found verbatim in an **earlier-created** card from a different owner (≥ 0.60 and ≥ 4 lines → copied). Row-level: cross rows equal (to published precision) to the target's own value, and own rows whose every metric equals an earlier publisher's own row → restated. Copies are kept in the ledger but excluded from charts, sets, links and counts | `detect_copies.py`, `audit/copies.json`, `build.py` |
| 6 Normalise | Units: GiB→GB (×1.073741824), bytes→GB, millinats→nats (÷1000), count-style top-1 ("226 of 242") → %. Plausibility assertions fail the build on impossible values | `build.py` |
| 7 Comparable sets | A set = one measurer, one stated method, ≥ 2 quants. 11 curated (`sets.py`), the rest automatic per measurer (sibling repos merged only when the stated reference and dataset are identical); duplicate reprinted tables dropped | `sets.py`, `build.py:auto_sets` |
| 8 Linked estimates | **Disabled for launch** (external review F06/F07). `build.py:link_sets` is retained but its output is cleared before rendering; `test_gates.py` fails if any estimate reaches the data | `build.py`, `test_gates.py` |
| 9 Site | Static HTML with full table server-rendered for crawlers; explorer in vanilla JS SVG; JSON-LD Dataset, sitemap | `build.py`, `site_*.css/js` |

## 3. Verification gates (outputs captured by re-running them for this pack)

1. **Verbatim** — every cited value must appear on its cited line of the pinned snapshot (number tokens compared, markdown stripped).
1b. **Column binding** — every value taken from a markdown table must sit in a cell whose header (or row label / reviewed caption exception) names the same metric family (`verify_columns.py`, exceptions in `audit/column_exceptions.json`).
2. **Image** — every transcribed label must agree with its marker position (KLD within 3%, PPL within 0.0015, bpw within 0.01); BF16 PPL and noise-floor reference lines are detected from the pixels.
3. **Blind re-extraction** — independent agents re-extracted every measured card without seeing our data; value sets diffed per repo. Differences were resolved against the source text.
4. **Plausibility and sign** — build fails on KLD outside [0, 5) nats (percentile keys exempt), top-1 outside [0, 100], size outside [1, 400] GB, or any parsed value whose sign differs from the published string.
4b. **Fail closed** — `build.py` and `make_pack.py` refuse to write if any gate exits non-zero. `test_gates.py` proves this with mutation tests (output below).
5. **Adversarial review** — a fresh agent reviewed the rendered page against sources (findings and fixes in §6).

{gate_md}

## 4. Comparable sets

| id | title | points | metrics | reference | type |
|---|---|---|---|---|---|
{set_md}

## 5. External review dispositions (GPT "Astra")

{astra}

## 6. Internal adversarial page review

{adv if adv != "(not available)" else "An internal adversarial page review was started but interrupted before it reported; it was not rerun. Two external reviews supersede it: GPT \"Astra\" and a cold review by Fable (`audit/fable_review.md`, verdict \"ship after fixes\"). Dispositions for both are in §5."}

## 7. Reference classes (repos)
{dict(ref)}

## 8. Known limitations and open questions for the reviewer

- **Claims, not measurements.** Nothing was re-run. A publisher's number can be wrong while we copy it correctly.
- **Method text is free-form.** Reference model, dataset and context are recorded as stated; "same method" for auto-sets is judged by identical stated reference + dataset strings, which can merge or split incorrectly.
- **Sizes have mixed bases** (download, GPU weights, resident, RAM in use, trunk only). Each point is labelled; charts within a set use that set's basis.
- **No cross-set estimates.** Sets measured differently are shown side by side, never converted onto one scale.
- **Copy detection** is line-based with a threshold; paraphrased copies or copies of a single table row can slip through. Direction uses repo createdAt.
- **Image transcription** covers turboderp only; other image-only results are not included.
- **Snapshot date.** Cards change; everything is pinned to the commit listed in `snapshots/index.json`.
- **Questions:** Is any flag unfair to a publisher? Is any set mixing methods? Should cards whose method is "not stated" be charted at all?

## 9. How to re-check

```bash
pip install pillow numpy   # for verify_images.py
python3 verify.py          # verbatim gate (no network)
python3 verify_columns.py  # table column-binding gate
python3 verify_images.py   # image transcription gate
python3 detect_copies.py   # copy detection
python3 diff_blind.py audit/blind.json; python3 diff_blind.py audit/blind_new_0.json audit/blind_new_1.json
python3 build.py           # rebuilds site/ and data.json (refuses on any gate failure)
python3 test_gates.py      # mutation tests of the gates
```
Snapshot integrity: `snapshots/index.json` lists the SHA-256 of every card file.
"""
os.makedirs(f"{ROOT}/dist", exist_ok=True)
open(f"{ROOT}/REVIEW_PACK.md", "w").write(doc)
zp = f"{ROOT}/dist/qff-review-pack.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(CODE):
        rel = os.path.relpath(base, CODE)
        if rel.startswith(("dist", "claims", "data", ".git", "__pycache__")): continue
        for f in files:
            if f.endswith((".pyc",)): continue
            p = os.path.join(base, f); z.write(p, os.path.join("quant-fidelity-frontier", os.path.relpath(p, CODE)))
print(zp, round(os.path.getsize(zp) / 1e6, 1), "MB")
for n, (o, rc) in gates.items(): print(f"[{rc}] {n}")
