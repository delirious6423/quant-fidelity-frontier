# Prompt: build a Quant Claims Ledger for another model

Copy everything below the line into a new session. First fill in the three values in **Inputs**. Attach `quant-claims-review-pack.zip`, the finished Qwen3.8-Flash-Next project, as the template. Or, once it is published, point the session at the repo.

---

## Inputs
- **Base model (Hugging Face id):** `{ORG/MODEL}`, e.g. `meta-llama/Llama-5-70B`
- **Site:** add the model to the existing GitHub Pages site `delirious6423/quant-claims` as `/{model-slug}/`. Keep one shared template and data schema for every model, plus a model index page. Never fork a separate site per model.
- **Template:** the attached `quant-claims-review-pack.zip`, which contains the Qwen3.8-Flash-Next build. Reuse its code: `build.py`, `verify*.py`, `detect_copies.py`, `gen_auto.py`, `sets.py`, `highlights.py`, `site_*`, `test_*.py`, `make_pack.py`, `.github/workflows/verify.yml`. Change only what this model needs, and keep the code model-agnostic, with model-specific values moved into a per-model config.

## Goal
Build a public, web-searchable site comparing **every Hugging Face quantization of `{ORG/MODEL}`**. Use only each publisher's **own claims**: model cards, plus docs pages a card links to as its results. Nothing is re-measured. A reader must be able to see:
1. what each publisher claims, and how they measured it;
2. which claims are comparable, meaning the same measurer and the same method;
3. which quant is most efficient *within* one measurement;
4. where cards disagree.

The site must never imply a ranking the evidence does not support. Publishers' reputations are at stake, so wrong attribution is as serious as a wrong number.

**Done means:**
- every catalogued card has been read in full;
- every published value is checked mechanically against its pinned source line;
- two independent reviews (a blind re-extraction and a cold adversarial review by a different model) are dispositioned;
- all gates and test suites exit 0 in CI;
- I have signed off;
- only then, the site is published.

## How to work
- Open with 1–2 lines on how you deepened the task, then execute without waiting for approval. Ask me only for decisions that are expensive to reverse, such as publishing or deleting.
- Keep `PLAN.md` and `PROGRESS.md` in the repo so a fresh session can resume. Session usage limits will interrupt long runs, so checkpoint after every phase and every agent queue.
- Run independent work as parallel agents, in queues of about 50 cards each, with explicit file ownership per agent. Check usage before a large fan-out.
- Never grade your own work. Every "done" needs an outside check, and you must show its output.

## Fast path (use this; measurements in `FASTPATH.md`)
Run these stages in order; each one can be resumed:

1. `qcl_fetch.py` (or the `fetch.yml` GitHub Action).
2. `autoextract.py`.
3. `qcl_map.py tasks`, then **two independent runs** of Sonnet mapping agents, 3 cards each, as many in parallel as limits allow. The agents return JSON mappings only and never type values.
4. `qcl_adjudicate.py`: accept the values both runs agree on, and send only the disagreements to one resolver pass.
5. `qcl_map.py apply` → `verified/`.
6. Build, gates and tests.
7. A cold review scoped to the new data.

Phases 1–4 below describe what these tools do and the rules they enforce. Fall back to reading a card in full only when triage, the mapper or the resolver marks it unresolved, or when its results exist only as images.

## Phase 1 — Catalogue (one API call)
- List all repos tagged `base_model:quantized:{ORG/MODEL}` in a single Hugging Face API call. Record id, sha, createdAt, lastModified, downloads and tags in `cache/catalogue.json`.
- Note derivative repos (finetunes or merges re-quantized) separately. Keep them in the catalogue, flagged `weights_modified`; never mix them into charts silently.
- Include the base model's own official quants (e.g. the publisher's FP8 or INT4) as baseline points.

## Phase 2 — Local source ledger (fetch once, then work offline)
- For each repo, fetch the raw `README.md` **at its pinned commit sha**. Do not use summarizing web fetchers; they paraphrase numbers.
- Store each card under `cache/cards/`, with SHA-256 and sha recorded in `cache/manifest.json` and `snapshots/index.json`.
- Also store:
  - HF file listings, for sizes;
  - branch listings, for repos with one quant per branch (exl3 style);
  - docs pages a card links to as its results;
  - chart images that carry results.
- Throttle requests, and stop on HTTP 429. Resume from the manifest; never re-fetch something already stored.
- If the shell cannot reach huggingface.co, use the browser channel, with me signed in if needed. Write oversized results straight to files and chunk payloads so they never route through chat.

## Phase 3 — Extraction (every card, read in full)
- Write claims as Python data, `verified/part*.py`. Each repo is a `repo(...)` entry with `variants` (the repo's own files), `ladder` (the publisher's comparison builds), `cross` (other publishers' files measured on this card), `methodology` (reference, dataset, context, tool, line refs) and `flags`.
- Each entry stores:
  - the **exact published string** for every value;
  - `L` (and `L_extra`) line numbers;
  - `size` with an explicit `unit` (GB / GiB / bytes) and `basis` (download, GPU weights, resident, RAM in use, text weights, and so on);
  - `of` for cross entries;
  - `measured=True/False` wherever the card's wording leaves who ran the test open;
  - `row_label` when the card's label differs from our name;
  - `unit_L` when the unit is stated only in a caption.
- Extraction rules:
  - **Quoted is not measured.** If a card reprints a figure from elsewhere ("reported by", "from their card", "publisher-reported", "per X's numbers"), set `measured=False`. It renders as "quoted by X, not measured".
  - **Other-model results are not rows.** For example, a card showing figures "from the conversion of GLM 5.1". Put these in a repo flag, not in data.
  - **Unnamed targets are inferences.** If a card names a file (e.g. `UD-Q4_K_XL`) but not its repo, the build marks the attribution as inferred.
  - **Reference classification uses the whole stated reference.** BF16, full precision, unquantized, or the base repo id counts as "vs BF16". Negations ("no direct BF16"), components ("BF16 n-gram table"), other models, and Q8_0 references do not.
  - Every flag cites its line. Use neutral wording: never assert why two cards differ.
  - Record a card that has no quality claims as "read in full: no quality claims", with a reason. Record unread cards as "not yet reviewed", never as "no claims".
- Run agents over queues. Give each one only the local snapshots and the schema, plus Phase 4's gates to run on its own repos before reporting. Keep an acceptance list of known-hard cards (big pivoted tables, image-only results, multi-card shared tables) and confirm each is extracted.

## Phase 4 — Gates (fail closed)
`build.py` and `make_pack.py` must refuse to write anything if any gate fails. Required gates:
1. **Verbatim:** every number token of every value appears on its cited line. Normalise markdown, thousands separators, space or thin-space grouping, and leading `.05`.
2. **Sign:** a signed value has the same sign on the source line. Normalise U+2212, U+2013 and U+2012.
3. **Unit:** the entry's unit sits next to the size number on the line, in the column header, or in a cited caption. Bytes and GiB are converted exactly; a size with no unit fails the build.
4. **Column binding:** a table value sits under a header of its metric family (KLD, top-1, PPL, size, bpw…). For pivoted tables, the column must name the variant.
5. **Row label:** the row's label cells name the entry, so swapping two variant names fails.
6. **Quoted context:** any cross or ladder entry near "reported by / quoted / as published…" must carry an explicit `measured` field.
7. **Images:** transcribed chart values match marker positions, and reference lines (BF16, noise floor) are detected from the pixels.
8. **Plausibility:**
   - KLD within [0, 5) nats;
   - top-1 within [0, 100], with fractions such as `0.913` and counts such as "226 of 242" converted;
   - sizes within [1, 400] GB, except non-model artifacts (control vectors, adapters, MTP heads, sidecars);
   - parsed sign equals the published sign.
9. **Mutation tests** (`test_gates.py`): on a throwaway copy, plant each error class and prove the matching gate fails. Error classes: absent value, swapped KLD/PPL columns, pivoted column swap, sign flip, GiB→GB flip, variant-name swap, quoted entry without `measured`, size without unit, changed image baseline.

## Phase 5 — Copies and restatements
- **Card-level copies:** if a card's numeric claim lines are found in an *earlier-created* card (by createdAt) from another owner, share ≥ 0.6 over ≥ 4 lines, the card is copied. Name as the source the older card with the **largest** overlap. Exclude lines from the base model's own card. Keep a hand-reviewed overrides file for copies that line matching misses, such as a card republished with different numbers; each override needs a reason.
- **Row-level restatements:** a cross row equal to the target's own figure, to published precision, is a restatement. **Never apply this between two files of the same publisher.**
- Copied and restated rows stay in the ledger but leave every chart, set and count.

## Phase 6 — Comparable sets
- A set means one measurer, one stated method and one size basis, with at least 2 quants.
- **Curated sets** (`sets.py`) cover the important measurers.
- **Automatic sets** cover everyone else:
  - one per measurer, where sibling repos merge only when the stated reference and dataset strings are identical;
  - use the title repo's own size basis first;
  - include rival files the card's author ran in the same test;
  - exclude quoted, inferred, copied and other-model rows;
  - drop a set whose values largely repeat another set from the same owner;
  - fail if a set lacks its publisher's own quant.
- Every set id is unique.
- Flag reference builds (Q8_0 reference, BF16 run, official FP8 baseline, teacher) with `is_ref`. They are never ranked as "efficient".
- One "every claim together" view exists for orientation. It is marked **not comparable** and has no efficiency tools.
- **No cross-set estimates or equating.** Units convert exactly; methods do not. Say so on the page.

## Phase 7 — Site
- Build static HTML from one template, which produces both the claude.ai preview artifact and the Pages build.
- Server-render the full ledger table for crawlers. Add JSON-LD Dataset markup, a sitemap, robots.txt and a canonical URL.
- Use vanilla JS and SVG; no CDN beyond Google Fonts. Support both themes through tokens, work at phone width (390 px) with no horizontal page scroll, meet basic accessibility, and add `[hidden]{display:none!important}` globally.
- Page order:
  1. header with stats;
  2. "Read this first" rules: claims not measurements, methods differ, sizes mean different things;
  3. the **explorer**;
  4. head-to-heads (one measurer, several publishers);
  5. "Same file, different numbers";
  6. the ledger;
  7. publisher notes;
  8. no-claims list;
  9. method.
- The sticky section nav highlights the current section.
- Explorer features (all exist in the template):
  - a grouped set chooser, plus "find a quant in any set", which jumps to the set and highlights the quant (press `/` to focus it);
  - metric chips;
  - a quant picker: text filter, format-family chips (top 8, then "+N more"), tick/untick matches, reset, expand-to-full-width list, tri-state per repo;
  - a clickable legend, folded into a button above 16 repos;
  - points you can tap to pin a tooltip, with every metric and the source line;
  - linked hover across chart, list and table;
  - log/linear toggle;
  - Max size and Max/Min metric sliders, size-budget presets (≤ 24/32/48/64/96/128/192/256 GB), and a size-type selector on mixed views;
  - "Only efficient quants" (Pareto: no other shown quant is no larger and better) and an optional frontier line;
  - a sortable table with a **"Beaten by"** column and Copy as CSV;
  - "Copy link to view" via a query string, restored on Pages.
- Ledger features: multi-word search, format and reference filters, rival/quoted and modified-weights toggles, sorting, reset.
  - Chips: "measured by X", "quoted by X, not measured", "repo not named on card", "modified weights".
  - Entry flags appear under the row.
  - Main KLD/top-1/PPL columns use canonical metric families and show the published key when it differs.
  - Mark "PPL change" as "vs the card's baseline".
- Never rank across sets, assert a cause, or speculate about a publisher's method.

## Phase 8 — Verification (outside checks only)
1. **Blind re-extraction:** independent agents re-extract every measured card without seeing our data. Diff per repo (`diff_blind.py`) and resolve each difference against the source text.
2. **Gates and tests:** `verify.py`, `verify_columns.py`, `verify_images.py`, `test_gates.py`, plus:
   - `test_claims.py`, the build-level regressions;
   - `test_ui.py`, Playwright at 1366 and 390 px. It covers every control, no horizontal scroll, no JS errors, unique DOM ids, and an independent Pareto computation compared against the UI. Test **visibility**, not attributes.
3. **Screenshots:** take them in light, dark and phone views, and look at them yourself.
4. **Review pack:** `make_pack.py` reruns everything and writes `REVIEW_PACK.md` (pipeline, gate outputs, sets, limitations, dispositions) and a zip. Send me the zip.
5. **Cold adversarial review by a different model** (Fable, or another lab's model). Give it the pack and an adversarial brief:
   - sample at least 60 values against snapshots;
   - read at least 15 no-claims cards in full;
   - attack set construction, attribution, flags and gates;
   - try mutations;
   - load the UI.

   It writes `audit/<reviewer>_review.md`. Disposition **every** finding in `audit/review_fixes.md`, and add a regression check to `test_claims.py` for each one.
6. Rerun everything until all checks exit 0, then republish the preview.

## Phase 9 — Publish (only on my explicit go-ahead)
- Push to `delirious6423/quant-claims` and enable Pages from `docs/`.
- CI (`.github/workflows/verify.yml`) runs every gate and test suite and checks that `site/` equals `docs/`.
- Send me the URL. Add the model to the index page.

## Pitfalls from the first build (check for each one)
**Fetching and parsing**
- Web fetchers paraphrase numbers. Use raw snapshots.
- Bulk fetches hit 429s. The ledger exists for this, so fetch once.
- Number parsing breaks on: unicode minus; `.0549`; `1 234` and `1,234`; "226 of 242"; top-1 written as a fraction; byte counts read as GB; GiB without conversion; millinats.

**Data and attribution**
- One table shared across several cards from the same publisher must use one size basis everywhere.
- Copy direction uses createdAt, not lastModified. The copy source is the largest overlap, not the oldest line.
- Matching an owner name against card text must not use generic short names ("GGUF" appears everywhere).
- Reference builds appear on the efficiency frontier unless flagged.
- Same-owner restatement rules delete real points.
- A majority-basis filter can drop the title repo's own quant.

**UI and tests**
- Duplicate auto-set ids make sets impossible to open.
- A CSS `display` rule silently overrides `hidden`.
- A test that reuses a variable name can mask a failure. Read the test log, not just the exit code.

**Process**
- Agents hit session limits mid-task. Resume them with a message instead of restarting, and keep file ownership clear while they run.

## Deliverables
1. The site, previewed as a claude.ai artifact and published to Pages after sign-off.
2. `site/data.json`, the dataset.
3. The review pack zip.
4. `audit/` with every review and its dispositions.
5. Updated `PLAN.md` and `PROGRESS.md`.
6. A short final report: counts (repos, read, with claims, vs BF16, rival measurements, quoted, copies), what the reviews found, what remains deferred, and the URL.
