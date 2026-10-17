# Dispositions — external review (GPT "Astra", 17 Oct 2026, quant-claims-detailed-review.pdf)

Each finding: status, what changed, where, and how it is now checked. "Test" refers to `test_gates.py`.

| ID | Finding | Status | Fix | Check |
|---|---|---|---|---|
| F01 | 257 leftovers declared claim-free without being read | **Accepted** | `gen_auto.py` now labels leftovers "Not yet reviewed"; site separates "read in full: no quality claims" from "not yet reviewed"; all 257 cards re-read by five agents (`verified/part5_*.py`) with Astra's 25 as an acceptance list (`audit/astra_missed25.json`) | Acceptance list check in §Acceptance below |
| F02 | Unicode minus dropped → 14 sign errors | **Accepted** | `build.py:num()/err()` normalise U+2212/U+2013/U+2012/U+2014/U+FE63/U+FF0D before parsing; build asserts raw-vs-parsed sign agreement | Test "unicode minus keeps sign"; build assertion |
| F03 | Failed verifier does not stop build/pack; number presence ≠ column binding | **Accepted** | `build.py` runs verify.py, verify_columns.py, verify_images.py first and refuses to write on any non-zero exit; `make_pack.py` refuses to pack on gate failure (no more grep pipe); new `verify_columns.py` binds table values to a header of the right metric family, with reviewed exceptions in `audit/column_exceptions.json` (7, each citing the caption line) | Tests "absent value blocks build and pack", "swapped KLD/PPL columns caught" |
| F04 | One source treated as one experiment (ROCmFP4 + AesSedai quoted rows) | **Accepted** | Auto-sets use only the measurer's own and ladder rows; quoted rival rows never enter an auto set | Test "auto sets: one basis, no quoted cross rows" |
| F05 | Mixed size quantities on one axis | **Accepted** | Auto-sets keep one size basis (majority basis; other-basis points dropped from that chart, still in ledger); TheDrainFlorist sizes relabelled "text weights (excl. vision, MTP)" | Same test |
| F06 | Restated values became anchors | **Accepted** | Restatement detection compares canonical metric families (kld≡mean_kld, top1≡top1_pct); linked estimates removed from the published site | Test "linked estimates disabled" |
| F07 | LOO bands presented as worst-case bounds | **Accepted** | Estimates disabled for launch; the method is documented as experimental, not shipped | Same test |
| F08 | Ledger drops 493 metric fields | **Accepted** | Ledger renders every metric: five main columns plus "Other metrics (as published)" containing all remaining keys | Test "no invisible metrics in ledger" |
| F09 | Metadata contradictions (BF16 badges, MXFP4→NVFP4, modified cross rows) | **Accepted** | `ref_class` rewritten: BF16 only when the stated reference is the BF16/full-precision model; "n-gram table", "no BF16 run", etc. → other; reviewed overrides file supported. `gen_auto` checks MXFP4/MXFP8 before FP4. Cross rows inherit the target repo's modified-weights status | jamesrogers / VnimanieAI / aniolekx now "other"; MXFP4 repos labelled MXFP4 |
| F10 | Publisher notes misstate evidence | **Accepted** | doth4580: loss comparison attributed to its 3.05 pack; bit-exact tensor check stated separately. TheDrainFlorist: "2 GiB larger" contradiction flag deleted; "104 GB" flag replaced by an explanation (text weights + 0.84 GiB vision + 2.1 GiB MTP); prose-only top-1/PPL noted. Intel: "saturated/easy" replaced by task list. AP/Gyro and disagreements intro: cause of differences no longer asserted. Missed curated claims added: Gyro code KLD, Jundot previous-build KL, Strata NVIDIA comparator KL 0.038, VQ bf16-teacher baseline | Verbatim gate on all new values |
| F11 | Audit completion not substantiated | **Accepted** | This file; blind-diff dispositions below; image baseline now checked (`verify_images.py` reference-line detection); copy-only repos excluded from the "publish quality numbers" count | Test "image baseline checked" |
| F12 | Explorer duplicate IDs; None leaves estimates; flag-search placeholder | **Accepted** | Checkbox ids unique per set and point; estimates removed (so All/None act on one list); ledger search placeholder corrected to what it searches | Test "unique explorer ids" |

## Blind-diff dispositions
- First 44 repos: 627 of our values; 617 independently found by the blind auditor; 7 are HF-listing sizes (checked by `verify.py` against the listing); 1 (primitive-ai 83.8, L267) present on the card, missed by the auditor; 1 token artifact ("8" from the label "FP8").
- Later 152 repos: 2,002 of our values; 1,957 independently found; the remaining 45 were inspected individually: confidence-interval bounds, standard errors, pass counts, completion-token counts, draft-acceptance rates and digits inside labels (e.g. "MMLU (200)") — all verbatim on their cited lines; none feed a chart.
- Auditor-only values are numbers we deliberately do not chart (component sizes, deltas restated in prose, hardware RAM); they are not disagreements.

## Not adopted / deferred
- A full artifact/experiment/metric schema (Astra §09) is the right long-term contract; this release keeps the current per-entry schema plus the gates above. Deferred to the next model.
- Cross-publisher estimates will return only with family-diverse anchors, extrapolation flags and external hold-outs (Astra F07 recommendation).

## Acceptance (after card re-read)
- All 453 cards read in full; `gen_auto.py` now catalogues 0 repos as "not yet reviewed".
- Astra's 25 missed cards: **25 of 25** have their claims extracted (checked against `site/data.json` rows with metrics).
- Repos publishing a quality number about their own files: 152 → 197. Read in full with no quality number: 256.
- New reviewed column exceptions: 2 for wrldsuksgo2mars (table headers name the reference build; the caption on the line before the table states every cell is perplexity).
- Plausibility gate: a size under 1 GB is now allowed only for artifacts that are not model files (control vector, LoRA/adapter, MTP/draft head, sidecar or n-gram table), identified from the format or variant name. Triggered by a 483,520-byte refusal-direction vector, which is stated correctly on its card. Byte sizes under 0.1 GB keep two significant figures instead of rounding to 0.0.

## UI changes (user request, 17 Oct)
- Explorer: quant filter (text and format chips) narrows both the list and the chart; tick/untick matches; reset; a clickable legend per repo; tap or click a point to pin a tooltip with its source line; hovering links the chart, list and table; a sortable table with Copy as CSV; an optional efficient frontier within one set; search across every set, which jumps to the set and highlights the quant; the chart is redrawn at the container width (phone layout); set ids made unique (two pairs of automatic sets shared an id, so the second of each could not be opened).
- Page: sticky section nav; publisher-notes search with a flags-only filter; no-claims status filter; ledger multi-word search, reset and text-column sorting; horizontal page overflow on phones fixed (a non-wrapping select and chips).
- Checked by `test_ui.py` (48 Playwright assertions at 1366 px and 390 px, including no horizontal scroll and no JS errors).

---

# Dispositions — cold external review (Fable, 17 Oct 2026, `audit/fable_review.md`)

Verdict received: **ship after fixes**; 0 numeric transcription errors in ~470 re-read values. Regression checks: `test_claims.py` (one or more per finding, run on the built site), `test_gates.py` (gate mutations), `test_ui.py` (explorer behaviour).

| ID | Sev. | Status | Fix | Check |
|---|---|---|---|---|
| F1 | blocker | **Accepted** | Entry-level `flags` and `measured` now reach the build. Rows the card quotes from elsewhere (`measured=False`, a "quoted / not measured / reported by" flag, or "quoted"/"-reported" in the name) become `kind="quoted"`: chip "quoted by X, not measured", excluded from every set, the head-to-head tables, the rival-measurement count and "Also measured in". Entry flags are printed under the row. The AesSedai IQ4_XS / agentionai pair is removed from "Same file, different numbers" (that card reprints AesSedai's own ratios, L163-165). A gate now requires an explicit `measured` field wherever the cited context says the figure is quoted | `test_claims.py` F1 (×36), `test_gates.py` quoted-context mutation |
| F2 | blocker | **Accepted** | Units added to the three ji-farthing entries (GiB); TheDrainFlorist's unit/basis special case moved into the data; ddalcu 73 GB given its unit. The build refuses any size without a unit; `verify.py` checks the unit token next to every size | `test_claims.py` F2, `test_gates.py` unit mutations |
| F3 | blocker | **Accepted** | `ref_class` reads the whole stated reference: BF16 / bfloat16 / FP32 / full precision / unquantised / the base repo id → "vs BF16", unless negated ("no direct BF16", "not Qwen BF16"), a component ("BF16 table", "BF16 PLE"), another model (GLM) or a Q8_0 reference. 23 repos changed class; all listed with their stated reference in `audit/ref_class_changes.json` and reviewed by hand (four first-pass misclassifications corrected) | `test_claims.py` F3 (positive and negative lists) |
| F4 | blocker | **Accepted** | Estimate sentence deleted; blind re-extraction sentence now says it covered the 196 repos with measured claims at the first pass and that the rest were read by one reader | `test_claims.py` F4 (forbidden strings) |
| F5 | major | **Accepted** | Restatement rule skipped when the cross row's measurer and target share an owner; Strata set now has all 10 rows incl. the NVFP4 + Q8_0 mix; duplicate-table test now keys on the share of repeated values, so the two Maximilian228 auto sets are gone | `test_claims.py` F5 |
| F6 | major | **Accepted** | Auto sets take the title repo's own size basis first. A set without its publisher's own quant fails the check (fuutott now has no auto set: its own row is on a different size basis from its ladder) | `test_claims.py` F6 |
| F7 | major | **Accepted** | Auto sets include rival files the card's author ran in the same test; quoted rows and rows whose target repo is not named on the card stay out; the set note says so | `test_claims.py` F7 |
| F8 | major | **Accepted** | Rows flagged "not this model" (inferencerlabs GLM 5.1 table, 7 rows) are not emitted; the repo flag explains why | `test_claims.py` F8 |
| F9 | major | **Accepted** | showkieo marked as a copy of beamster by hand (`audit/copy_overrides.json`, with reason); its set removed; both flags reworded (no "disagree") | `test_claims.py` F9 |
| F10 | major | **Accepted** | The build marks a cross row `inferred` when the measuring card never names the target's owner (turboderp ×5, yamz-labs ×2, Jakevin, matteiuspi). Shown as "repo not named on card" in the ledger, table and tooltip; such rows are not cross-linked and do not enter auto sets | `test_claims.py` F10, `test_ui.py` |
| F11 | major | **Accepted** | `verify.py`: signed tokens must match the source (97 checked); every size needs its unit beside the number or in a cited caption (789 checked). `verify_columns.py`: row-label and pivoted-column binding (974 checked, 245 pivoted). Quoted-context gate requires an explicit `measured` field. 49 `row_label` and 13 `unit_L` fields added from the cards' own wording, 14 reviewed exceptions; no number changed; 0 failures. Six mutations added to `test_gates.py` (14/14 pass) | `test_gates.py` |
| F12 | major | **Accepted** | `detect_copies.py` chooses the older card with the largest overlap. Kashapps now names kingjones777 Uncensored-FAST (21/21, base-card lines excluded). Three repos newly cross the copy threshold: vaultai ← TensorFold (11/12), auto7x ← jpezzulli (0.71), ddalcu MLX-Serve-8bit ← airagrp (0.71) | `test_claims.py` F12 |
| F13 | major | **Accepted** | Reference builds (Q8_0 reference, FP8 baseline, BF16 runs, "teacher") flagged `is_ref`; never on the frontier, never a "Beaten by" entry, shown as "reference: not ranked" | `test_claims.py` F13, `test_ui.py` |
| F14 | major | **Accepted** | Note now: "Method not published; Q8_0 is reported at KLD 0.0266 and 94.1% top-1." | `test_claims.py` F14 |
| F15 | major | **Accepted** | "Measurements of rivals' files" counts measured, uncopied rival rows only (186 → 152; 23 rows are now marked quoted) | `test_claims.py` F15 |
| F16 | minor | Accepted | `hf_branch_total` label added | `test_claims.py` F16 |
| F17 | minor | Accepted | Column header "PPL change — vs the card's baseline", with a tooltip | `test_claims.py` F17 |
| F18 | minor | Accepted | TheDrainFlorist's shared table labelled "text weights" on every card | `test_claims.py` F18 |
| F19 | minor | Accepted | Removed with the F1 pair | `test_claims.py` F19 |
| F20 | minor | Accepted | Frontier and efficient-only controls hidden on the not-comparable set; top-1 written as a fraction (0.913) converted to % with a note | `test_ui.py` |
| F21 | minor | Accepted | Main KLD / top-1 / PPL columns use the canonical families (incl. `kl`, `mean_kl`, `kld:english`); the published key is shown under the value | `test_claims.py` F21 |
| F22 | minor | Accepted | Follows from F5 (both views now show 10) | `test_claims.py` F5 |
| F23 | minor | **Deferred** | Flags without a line citation are kept; adding citations to every flag is scheduled for the next model's pass | — |
| F24 | minor | Accepted | ISTA baseline size cites L123 and the build checks the number is on that line | `test_claims.py` F24 |


Known limitation left open: umixer's "uniform oQ4e" cross row combines a size quoted from Jundot's card (L85) with a memory figure umixer measured (L62); it is shown as a measurement with the size's source on the card.
