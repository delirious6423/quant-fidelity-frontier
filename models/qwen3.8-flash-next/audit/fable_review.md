# Cold adversarial review — Quant Claims Ledger (Fable, 2026-10-17)

Scope: `/home/claude/quantcmp` as found on disk (build.py / site rebuilt at 19:19 while this review ran; only UI additions — range sliders, "Only efficient quants", "Expand list" — changed; nothing below depends on those edits). All checks were made offline against `snapshots/` and `cache/cards/` (SHA-256 of all 453 card files re-computed and matched `snapshots/index.json`). Gates re-run: `verify.py` 3506/0, `verify_columns.py` 1391/0, `verify_images.py` 14/0, `test_gates.py` 8/8, `test_ui.py` 48/48.

## Verdict: **ship after fixes**

I did not find a single transcribed number that differs from its cited snapshot line (about 450 values re-read, table at the end). The numbers are right. What is not right is what the site *says about* some of those numbers: who measured them, which model they describe, what unit they are in, and which reference class they belong to. Several of these put a wrong statement next to a named publisher, which is exactly the class of error the project set out to avoid, so they should be fixed before the page is public. None of them requires re-architecting; most are 1–20 line changes in `build.py`, `sets.py`, two `verified/` entries and one flag.

Blockers: F1, F2, F3, F4. Majors: F5–F15. Minors: F16–F24.

---

## Findings

### F1 — BLOCKER — Rows that the card itself says are *quoted* are rendered as "measured by <publisher>"; extractor qualifiers are silently dropped
**Where:** `build.py:build_rows()` (`add()` never reads `e.get("flags")`); ledger chip `measured by …` (`render()`), explorer table chip (`site_explorer.js:293`), head-to-head table (`highlights.py`).

**Evidence (three independent instances):**

1. `mensaprodigy/Qwen3.8-Flash-Next-REAP-384-mlx-mtp` cross rows (`of` sh0wie …, HumanEval 93.9 / 92.1 / 91.5 / 88.4 %). Snapshot L125–L127: `## REAP-lineage evaluation (not this checkpoint)` … `The figures below are reported by Sh0wie for the related REAP pruning lineage`. The ledger renders them as `sh0wie/REAP-384-MLX-4bit  measured by mensaprodigy`. The restatement detector does not catch them because the variant strings differ from sh0wie's own rows ("REAP-384 Q4" vs "REAP-384 4-bit") and two of the `of` values are not repo ids. Same pattern: `mensaprodigy/…-REAP-288-oQ4e…` (`…(publisher-reported, quoted)`), `jomcgi-org/…-NVFP4-oominf` (`NVFP4 on SGLang (RadixArk-reported, quoted)`).
2. Extractors attached `flags=["quoted, not measured"]` to 18 cross/ladder entries (Wladastic ×3, intel-ai ×4, umixer, zankich, sleepyeldrazi, matteiuspi, …). `build.py` ignores entry-level `flags`, so e.g. `sleepyeldrazi … IQ3_XXS (base pack) ppl:wikitext 3.9044` (snapshot L62: `reports 3.9044 for the base IQ3_XXS pack … if you want a reference point`, i.e. ISTA's figure) is shown as *measured by sleepyeldrazi*, and `zankich … omlx 4-bit` likewise.
3. Head-to-head "Same file, different numbers" → row *AesSedai IQ4_XS*: second line reads `agentionai | PPL +3.12% | wikitext-2, ctx 2,048; quotes 117.13 GiB, an older build`. Snapshot `agentionai__…ROCmFP4-FAST-imatrix-GGUF` L163–L165: *"Against AesSedai's quants, compared the fair way (each build's PPL against its own measured reference, since **their test methodology differs from ours**)"*. agentionai did not run AesSedai's file at ctx 2,048; it reprinted AesSedai's own ratio. The table therefore presents a non-existent second measurement as a "disagreement", attributes a setup to it that the card denies, and the row inherits `ref_class=bf16` from agentionai. This is the exact F04 case the previous review raised; it was fixed for auto-sets but not for the ledger or the head-to-head table.

**Why it matters:** "Measurements of rivals' files: 186" (header stat) and every `measured by X` chip is a factual statement about X. For the cases above it is false.

**Fix:** (a) render entry-level `flags`/`note` and, when they contain "quoted"/"reported"/"not measured", set `kind="quoted"` (new chip "quoted by X", excluded from the cross count and from highlights); (b) in `highlights.py` drop the AesSedai/agentionai pair or relabel it "quoted by agentionai from AesSedai's earlier card"; (c) extend the restatement detector to compare by `of`-repo + canonical metric value when the variant string differs; (d) add a gate: any cross entry whose cited line (±3 lines) contains "reported by|quoted|per their card|as published" must carry an explicit `measured=True/False` field.

### F2 — BLOCKER — GiB sizes stored without a unit are plotted as GB (3 values, two sets)
**Where:** `verified/part1.py` ji-farthing ladder `IQ4_KT_v2 (…_v2 repo)` size=80.72 (no `unit`); `verified/part4_*.py` ji-farthing `_v2` ladder `Previous IQ3_KT…` 80.22 and `Previous IQ4_KT…` 88.42 (no `unit`). `build.py:to_gb()` treats a missing unit as GB.

**Evidence:** snapshot `ji-farthing__…ik-llama-GGUF@8fb1860` L39: `| [IQ4_KT_v2](…) | 80.72 GiB | …`; `…GGUF_v2@ca863d1` L71: `| Previous IQ3_KT with IQ4_NL fallbacks | 80.22 GiB | …`, L72: `| Previous IQ4_KT with IQ4_NL fallbacks | 88.42 GiB | …`. Rendered: curated set `jifarthing` x-axis "File size, GB (converted from GiB)" has IQ4_KT_v2 at **80.72** while the same file's own row (v2 repo) is at 86.67; auto set `a-ji-farthing-ik-llama-gguf-v2` is labelled "Size, GB (unknown)" and plots 80.22/88.42 (true 86.1/94.9). Ledger "Size is" column is blank for these rows.

**Fix:** add `unit="GiB"` to the three entries; in `build.py` refuse to build when `size` is set and `unit` is None (there are 11 such entries; 7 are TheDrainFlorist, already patched by a special case that should become data, 1 is ddalcu `73 GB`). Add to `verify.py`: when the cited line contains "GiB" within 12 characters of the size token, the entry's unit must be GiB.

### F3 — BLOCKER — `ref_class` labels ~15 publishers "other or no reference" although their stated reference is the BF16 model; the headline "Of those, against BF16: 74" is undercounted and the "vs BF16" filter hides them
**Where:** `build.py:ref_class()` regex `^(own |the |exact-checkpoint |local |hf )?(bf16|fp32|fp16|full[- ]precision|unquantized|original)`.

**Evidence (stated reference → class):** `scorpoon/…-exl3-SC` "Qwen3.8-Flash-Next BF16" → other; `pipenetwork/…` ×4 "bfloat16 (upstream)…" → other; `perkel/…-MC` "Qwen's BF16 weights computed with FP32 math" → other; `ivanfioravanti/…-DS4-Q4` "exact-checkpoint local BF16" → other (two prefixes); `OsaurusAI/Allosaurus…` and `JANGQ-AI/…JANGH4` "full bf16 model run layer by layer" → other; `amd/…-AttnFP8-PLEFP8` "Qwen/Qwen3.8-Flash-Next (BF16)" → other; `amd/…-PLEFP8` "Qwen/Qwen3.8-Flash-Next (own run)" → other; `intel-ai/…` ×3 "Qwen/Qwen3.8-Flash-Next bf16 (own run)" → other; `RedHatAI/…NVFP4` "Qwen/Qwen3.8-Flash-Next (unquantized)" → other; `IHaveNoClueAndIMustPost/…` "full BF16" → other; `manuuuel/mica…` "authors' unquantised weights" → other. The ledger badge on e.g. scorpoon's rows reads `other or no reference` (checked in the rendered page) while turboderp's identical-method rows read `vs BF16`.

**Fix:** classify from the whole string, not the head: `bf16|bfloat16|fp32|full[- ]precision|unquantis?z?ed|original model|Qwen/Qwen3\.8-Flash-Next(?!-FP8)` → bf16 unless preceded by "not |no |n-gram|table". Add a one-line review file listing every repo with quality claims and its class, and a test that the class of a stated-BF16 reference is never "other".

### F4 — BLOCKER — The Method section makes two statements about the site's own process that are not true
**Where:** `build.py:render()` method paragraph (rendered in `site/index.html` §"How this was built").

**Evidence:** (1) *"Estimated points in the explorer come from fitting two measurement sets … and are always shown as estimates with their error."* — linked estimates are disabled (`for s_ in sets: s_['linked'] = []`, `test_gates: linked estimates disabled`); no estimate exists on the page. (2) *"A second reader re-extracted every card without seeing our data"* — the blind re-extraction covered the 44 + 152 measured repos (`audit/blind*.json`, REVIEW_PACK §3); the 257 `part5_*` cards were read by one agent each and have no blind counterpart.

**Fix:** delete sentence (1); change (2) to "every card with measured claims at the time (196 of 453)". Add the sentence to `test_gates.py` as a forbidden string while estimates are off.

### F5 — MAJOR — Same-owner "restatement" rule deletes a real point from the Maximilian228 Strata set; the dedupe then fails and the same table appears as a second, partial set
**Where:** `build.py:build_rows()` cross-row restatement loop; `build_sets()` dup test.

**Evidence:** `Maximilian228/…-NVFP4-GPTQ-Strata` L34 `| NVFP4 + Q8_0 mix (ours) | 95.0% | … | 105 GiB |` is one row of the same 5,153-token test as the other 8 rows (L33–L42). Because the Mix repo's own card repeats 95.0 %, the row gets `copied_from=<itself>` and is dropped from the curated `strata` set, so the set shows the GPTQ build (93.6 %) as the best Maximilian228 quant and the frontier runs GPTQ→Q8_0. With 95.0 missing, the duplicate test (`repeats(o) >= 3`) sees only two repeated values and admits auto set `a-maximilian228-q8-0-strata`, which is the same table reprinted on the Q8_0 card (3 points, none of the rivals). The explorer's "find nvfp4" now lists `NVFP4 GPTQ` twice as if measured twice.

**Fix:** skip the restatement rule when `owner(measured_by) == owner(of)`; drop the Q8_0-Strata and NVFP4-Q8-Mix auto sets (the curated set already covers the table) or key the dup test on value multiset rather than qid overlap.

### F6 — MAJOR — Auto sets named after a repo do not contain that repo's quant (majority-basis filter)
**Where:** `build.py:auto_sets()` `pts = [p for p in pts if p["basis"] == main_basis]`.

**Evidence:** `a-maximilian228-nvfp4-q8-mix-strata` ("Maximilian228 · NVFP4-Q8-Mix-Strata") contains only Q8_0 (100 %) and NVFP4 GPTQ (93.6 %); the Mix quant itself (own row, 171.6 GB *disk*) is dropped because the two ladder rows are *RAM in use*, although L41 gives 105 GiB RAM-in-use for it. `a-fuutott-mlx-t5-imatrix-q3down-ple8` contains three experimental ladder builds and not the published checkpoint: its own row was filed with the 97,059,902,320-byte *disk* size from L63 although L107 (the same table row as the ladder entries) gives `35.6 GiB` resident. A reader opening "fuutott · …" sees three builds, none of which is the repo.

**Fix:** when a row carries sizes on more than one basis, store both (`sizes={basis: (v, unit)}`) and let the set pick the basis; at minimum, fail the build when a set's title repo has no point in the set.

### F7 — MAJOR — Auto sets exclude the measurer's own same-test measurements of rival files, while curated sets include them; the result favours publishers' own builds
**Where:** `build.py:auto_sets()` `if r["kind"] not in ("own", "ladder"): continue` (F04 fix).

**Evidence:** `beamster/…-Sushi-2bpw` L43–L44: *"every pack scored by sushi"*, table L48–L56 contains oMLX oQ5e, mlx-serve mixed-4-8bit, oQ4e, iQ-MLX 3.3bpw, Vontra 4-bit — all run by beamster in one sitting. The set `a-beamster-sushi-2bpw` shows the four Sushi packs only. The previous review's F04 concerned *quoted* rows (AesSedai in the ROCmFP4 card), not rows the measurer ran itself; the fix removed both. Gyro/AP/Strata (curated) keep such rows, so the two set types are built on different rules without saying so.

**Fix:** carry a per-entry `measured` flag (see F1) and admit cross rows with `measured=True` to auto sets; otherwise state in each auto set's note that rival rows from the same table were omitted.

### F8 — MAJOR — Ladder rows measured on a different model (GLM 5.1) are listed as Qwen3.8-Flash-Next quality rows
**Where:** `verified/part5_0.py` `inferencerlabs/Qwen3.8-Flash-Next-MLX-Q4i` (7 ladder rows) and the sibling `…-MLX-Q9`.

**Evidence:** snapshot L27: *"Due to system memory and time constraints, the figures shown below are from the conversion of GLM 5.1."* The ledger shows `inferencerlabs/MLX-Q4i  publisher comparison build | Q3 (GLM 5.1 conversion) | PPL 168.0`, `Q4-INF … PPL 1.21093, token_accuracy 97.70 %` etc. in the PPL column of a Qwen comparison. The extractor attached `flags=["measured on GLM 5.1, not this model"]` to each row, which `build.py` drops (F1).

**Fix:** move these to the repo's flags as prose and do not emit rows for them; generally, refuse rows whose entry flags contain "not this model".

### F9 — MAJOR — `showkieo/…-Sushi-3bpw` is treated as an independent measurer, and beamster's card carries a "the two cards disagree" flag against it
**Where:** auto set `a-showkieo-sushi-3bpw`; flags on `beamster/…-Sushi-3bpw` and `showkieo/…-Sushi-3bpw`.

**Evidence:** showkieo's card (created 2026-09-27, three days after beamster's) is beamster's Sushi-3bpw card text: L44 `hf download beamster/Qwen3.8-Flash-Next-Sushi-3bpw …`, L144 `hf download beamster/Qwen3.8-Flash-Next-Sushi-4bpw ngram_table.bin`; the quality table (L113–L120) bolds Sushi-4bpw and Sushi-3bpw as "ours" although both repos are beamster's. The numbers differ from beamster's *current* card because beamster re-measured later; `detect_copies.py` misses it because it compares against the current beamster snapshot. The site nonetheless (a) gives showkieo its own comparable set, (b) labels beamster's Sushi-4bpw a showkieo "publisher comparison build", and (c) flags beamster's card: *"showkieo … card prints different numbers for the same packs … the two cards disagree"*. A mirror of your own earlier card is not a disagreeing measurement.

**Fix:** mark showkieo as a copy (manual override in `audit/copies.json`), remove its set, and reword both flags to "showkieo's card is an earlier revision of this card (download commands point to beamster); its numbers are superseded".

### F10 — MAJOR — Measurements attributed to `unsloth/Qwen3.8-Flash-Next-GGUF` where the card never names unsloth
**Where:** `verified/part3_images.py` (turboderp, 5 cross rows), `verified/part4_4.py` yamz-labs (2 cross rows), `Cyronius`, `matteiuspi`, `BoldingBuilds`, `nibor1896` (see checker output: 40 cross rows whose `of` owner does not appear within 40 lines of the cited line).

**Evidence:** turboderp's card contains no occurrence of "unsloth" (grep of the snapshot); the chart legend says "GGUF" and labels `UD-IQ1_M` … `UD-Q4_K_XL`. yamz-labs L56: *"Strata v0.1.40 (UD-IQ4_XS pack) and Gufo v0.8.0 (UD-Q4_K_XL pack)"*, no publisher. The ledger states `unsloth/GGUF measured by turboderp` as fact, and the explorer's "Also measured in" / find box propagate it ("UD-Q4_K_XL … turboderp · ExLlamaV3 quality benchmark"). The NVFP4 rows on the same chart were handled correctly ("checkpoint not named"). Several other UD-named files exist in the catalogue (`xautonomics/…UD-IQ4_XS.LAYR`, `Guile/…`, `SpeederX/…UD-Q2_K_XL-EM`), so the inference, while probable, is an inference.

**Fix:** `of="unsloth/… (inferred from UD- file names; not named on card)"` with `inferred=True`, rendered as "probably unsloth" and excluded from "Also measured in" links; or keep the attribution and show the inference in the tooltip and ledger chip.

### F11 — MAJOR — The gates do not detect four realistic error classes (mutation evidence)
**Where:** `verify.py`, `verify_columns.py`, `build.py` assertions.

**Evidence:** In a scratch copy (`/tmp/claude-0/…/scratchpad/mut`, project files untouched) I applied four mutations and re-ran `verify.py` and `verify_columns.py`; all four passed with `failures: 0` / `unexplained mismatches: 0`:
1. Pivoted table, variants in columns: swapped perkel MCTiny's held-out KL/top-1 (0.1454 / 86.7 %) with MCSmall's (0.0613 / 91.7 %) on the same row L66. `verify_columns.py` accepts any cell whose *row label* matches the metric family, so column identity is never checked for tables where variants are columns (Jundot, Dyluhn, Vemloir, Litwein, Baekpica, perkel, meldlabs, intel-ai, glyd, … ≈190 rows flagged by my row-label scan).
2. Sign flip: PaoAI `ppl_delta_pct "−0.16 %"` → `"+0.16 %"`. `check_value()` falls back to digit tokens, which ignore sign; the build's sign assertion only compares raw against parsed, never against the source.
3. Unit flip: AesSedai Q5_K_M `unit="GiB"` → `"GB"`. No gate reads the unit (this is how F2 got through).
4. Variant swap: exchanged the names of AesSedai Q5_K_M and Q4_K_M, keeping their `L`. No gate compares the entry name with the row label.

The REVIEW_PACK's "3506 values verified verbatim" therefore means "each number appears somewhere on the line", not "the row says what we say it says".

**Fix:** (a) `verify_columns.py`: when the header row contains the entry's variant name (pivoted table), require the value's column header to match the variant; (b) `verify.py`: when the raw string starts with a sign, require the same signed token on the line; require the unit token (GB/GiB/bytes) within 12 chars of a size; (c) add a row-label check: for table rows, at least one token of the variant name (or an explicit `row_label` field) must appear in the first two cells; (d) add these four mutations to `test_gates.py`.

### F12 — MAJOR — Auto-detected copy flag names the wrong source card
**Where:** `detect_copies.py` (`min(older, key=createdAt)` per line) → flag on `Kashapps/…-Uncensored-ROCmFP4-FAST-GGUF`.

**Evidence:** rendered flag: *"Card's numbers match kingjones777/…-ROCmFP4-**STRIX_LEAN**-GGUF's card (13 of 21 claim lines identical…)"*. Re-computed with the same `claim_lines()`: Kashapps shares **23 of 23** lines with `kingjones777/…-Uncensored-ROCmFP4-**FAST**-GGUF` (created 2026-08-27 15:04, before Kashapps 09-14) and 15/23 with STRIX_LEAN (created 51 minutes earlier). The algorithm attributes each shared line to the oldest card containing it, so a near-verbatim mirror is reported as a partial copy of the wrong repo.

**Fix:** choose the source as the older card with the *largest* overlap (ties → oldest), then report that overlap; re-run and diff `audit/copies.json`.

### F13 — MAJOR — "Efficient frontier" marks the *reference* build and will now drive "Only efficient quants"
**Where:** `site_explorer.js:frontier()`; new `#exonly` filter (build.py edit at 19:19).

**Evidence:** Strata set with frontier on: `Q8_0 (reference) frontier` (top-1 = 100 % by definition; it is the yardstick, not a candidate). In the nvidia set the FP8 baseline point is likewise eligible. With the new "Only efficient quants" filter the reference will be presented as the most efficient quant at its size.

**Fix:** exclude `kind === 'baseline'` and variants containing "(reference)" from the frontier; say in the chart note that the reference is excluded.

### F14 — MAJOR — The unsloth set note speculates about unsloth's method
**Where:** `sets.py` set `unsloth`, `note`.

**Evidence:** *"Q8_0 sits at KLD 0.0266 and 94.1 % top-1, which suggests the reference is not BF16 or the eval set is unusual."* The pack's F10 disposition says causes of differences are no longer asserted; this one still is, about the most-downloaded publisher. There are other explanations (long-context chunks, the n-gram table, a different tokenizer path) and the site has no evidence for any of them.

**Fix:** "Method not published; Q8_0 is reported at KLD 0.0266 and 94.1 % top-1."

### F15 — MAJOR — Header stat and ledger count quoted rows as "Measurements of rivals' files"
**Where:** `build.py:main()` `stats["cross"]`.

**Evidence:** 186 = every `kind == "cross"` row, including restated/copied rows (`copied_from` set), quoted rows (F1) and rows whose `of` is "checkpoint not named". After F1/F5 the honest number is lower; compute it as cross rows with `measured=True and not copied_from`.

### F16 — minor — Raw basis key leaks into the ledger
`BASIS_LABEL` has no entry for `hf_branch_total`; turboderp's five own rows show "Size is: hf_branch_total" in the ledger (seen in the rendered page when sorting by Mean KLD). Add `"hf_branch_total": "download (HF branch listing)"`.

### F17 — minor — "PPL change" column mixes baselines without saying so
Rows show `+0.1270 %` (AesSedai, vs BF16), `−5.9 %` (kingjones777 FAST-imatrix, vs its own non-imatrix sibling, L55–L58), `−0.16 %` (PaoAI, vs its previous build, L59–L60) under one header next to a reference badge that describes the *repo's* reference. Rename the header "PPL change (vs the row's stated baseline)" or put the baseline in the cell title.

### F18 — minor — One file, two size bases
`TheDrainFlorist/…-VQ-5.5bpw` ladder `affine q3 (ours)` 75 GiB is `basis="disk"` (rendered "download") in `part4_0.py` while the identical row from the 4.4bpw and 3.2bpw cards is "text weights (excl. vision, MTP)". Same number, two labels, visible together when the ledger is sorted by KLD.

### F19 — minor — Head-to-head "Setup" text states things the card does not
"AesSedai IQ4_XS … wikitext-2, ctx 512 (current build, 114.99 GiB)": the AesSedai card states no method (set `aessedai` says "Not stated on card (repo logs: …)"). Say "method not on card; repo logs: wikitext-2, ctx 512".

### F20 — minor — Explorer: "Efficient frontier" checkbox stays enabled on the non-comparable "all" set (no frontier is drawn, but the control suggests one could be); the `all` set still carries `top1` only from `top1_pct`, so publishers who wrote `top1` (unsloth, Ar3ntiz `0.913` fraction) are silently absent from the top-1 view. Disable the checkbox on `all`; document the key normalisation.

### F21 — minor — Ledger main columns only bind `mean_kld` / `top1_pct` / `ppl`
`kld:english` (ji-farthing), `kl` (yamz-labs), `mean_kl` (Maximilian228), `kld:E/W/K` (sirfyyn) land in "Other metrics", so "sort by Mean KLD" and the "Only rows with a quality number" filter treat them differently from identical metrics with another key; the same regex gap is why sirfyyn (two quants, KLD on three corpora) has no auto set. Extend `CANON`/`MAIN` or canonicalise keys at load time.

### F22 — minor — Highlights bar chart for Maximilian228 includes the Mix row that the explorer set excludes (F5), so the two views of the same table disagree on how many quants were tested (10 vs 9).

### F23 — minor — `beamster/…-Sushi-2bpw` flag "Same iQ-MLX 3.3bpw file: 86.28 % top-1 here vs 85.6 % on ddalcu's own card (different scorers)" is fine, but ddalcu's figure should be cited with its line; several flags cite no line.

### F24 — minor — `ista` baseline point `size_gb=354` is hard-coded in `sets.py`; it is on the card (L123 `| BF16 | 16.00 | 354 |`), so cite it (`L=123`) instead of typing it.

---

## What I checked and found correct

- Snapshot integrity: 453/453 SHA-256 match.
- Image transcription: all 24 printed labels on `turboderp_qb_kld.png` and `turboderp_qb_ppl.png` read by eye match `part3_images.py` (0.0684 … 0.0031; 1.4879 … 1.3825; GGUF 0.0804/0.0533/0.0349/0.0165/0.0084 and 1.5031/1.4619/1.4342/1.4017/1.3926; NVFP4 0.0100/0.0241 and 1.3970/1.4199); noise floor 0.00249 and HF BF16 1.38423 are printed on the charts.
- 18 `part5_*` cards marked "No quality claims (card read in full)" read in full: kingjones777 STRIX_LEAN, igorls mixed-NInfer, Guile, hstolte HALOLOOM, Quat3rnion Signal-HGN-v2, doth4580 EXL3-3.05bpw, xpashkin iQ-MLX-6.6bpw, kingjones777 MTP-Heads, butterf1ying REAP288, densohax heretic2, TensorFold MLX-8bit-MTP, AutomatosX AXQ-MXFP4, Pragmatism0220 abliterated-direction, Wladastic Ombro-v9, ArtomYuan ROCmFPX, Litwein REAP320 MTPLX, trustos GSQ-RCO-Q2-MTPLX, nmatteo3294 Q3_PLE-MTP. No missed quality number. Borderline items were already flagged by the extractor (igorls L289 fp8-KV smoke test; Wladastic per-layer "beats UD-Q4_K_XL" without figures; trustos MTP acceptance). A regex pre-screen of all 188 such cards for metric words next to numbers found nothing else.
- Flags: no abusive language; the questionable ones are F9 (beamster/showkieo), F12 (Kashapps source), F14 (unsloth note) and the generic "Card suggests disabling Strata's SHA256 verification" (fair, it is on the cards).
- UI (Playwright 1366 px): tooltips show the raw string and the converted GB with the GiB original; axis labels name the basis; the set selector says "not comparable" for `all`; the frontier note says "in this set only"; the ledger sorts mnats rows by their converted value (1312.4 mnats sorts above 0.69 nats, correct). No JS errors.

---

## Sampled values (every value re-read against its cited snapshot line)

Legend: PASS = number, metric, variant, sign, unit, measurer and reference all as rendered; FAIL = at least one of those wrong (reason given). "n values" counts the individual numbers checked on that row.

| # | Measurer → row (kind) | Cited line(s) | Values checked | Result |
|---|---|---|---|---|
| 1 | perkel/…-MC → Flash-Next-MCTiny (own) | L66–68, L50 | KL 0.1454/0.0760/0.2234, top-1 86.7/90.0/85.2 %, 65 GB | PASS (7) |
| 2 | perkel/…-MC → Flash-Next-MCSmall (own) | L66–68, L50 | 0.0613/0.0283/0.1260, 91.7/93.7/88.7 %, 80 GB | PASS (7) |
| 3 | Jundot/…-oQ4e-mtp → oQ4e (updated) (own) | L34–35, L53, L63 | kl:code 0.0776, kl:novel 0.1713, 4.724 bpw, 106.294 GB | PASS (4) |
| 4 | Jundot/…-oQ4e-mtp → previous 2026-08-27 (ladder) | L34–35 | 0.0877, 0.1909 | PASS (2) |
| 5 | agentionai Gyro → Gyro-S/M/L (own ×3) | L57–59 | 27.64/34.98/50.21 GiB, KLD 0.401/0.291/0.146, top-1 75.29/78.75/83.87 % | PASS (9) |
| 6 | agentionai Gyro → ISTA IQ2_XS, IQ3_XXS; unsloth UD-Q2_K_XL; AtomicChat AD-4.27; ggml-org IQ4_NL; AP-Q4_K_XL (cross ×6) | L60–65 | 36.52/43.80/46.63/52.26/68.21/67.38 GiB; 0.420/0.311/0.334/0.228/0.200/0.107; 74.22/77.68/76.61/80.58/81.29/85.52 % | PASS (18) |
| 7 | agentionai AP → AP-Q5_K_XL … AP-IQ2_S (own ×7) | L47–53 | download 112.51/103.57/94.20/88.04/84.24/80.72/76.03 GiB; KL 0.0839/0.0771/0.0886/0.1243/0.1584/0.1844/0.2657; top-1 86.63/86.95/86.14/84.22/82.89/81.85/79.14 % | PASS (21) |
| 8 | agentionai AP → AP-Q5_K_M, AP-Q4_K_XL code KL/top-1 (own) | L69–70 | 0.0181/97.69 %, 0.0206/97.38 % | PASS (4) |
| 9 | agentionai AP → unsloth UD-Q4_K_XL (cross) | L71 | 103.69 GiB, 0.0778, 87.21 %, 0.0192, 97.36 % | PASS (5) |
| 10 | Maximilian228 GPTQ-Strata → NVFP4 GPTQ (own) | L36 | 93.6 %, 69 GiB RAM | PASS (2) |
| 11 | Maximilian228 GPTQ-Strata → Q8_0 ref; UD-Q4_K_XL; NVIDIA NVFP4; UD-Q3_K_XL; ISTA IQ3_S/IQ3_XXS/IQ2_XS/Q2_0 (cross ×8) | L33, L35, L37–42 | 100/94.1/92.9/90.7/90.0/87.3/85.0/84.1 %; 106/77/69/58/52/45/38/37 GiB | PASS (16) |
| 12 | Maximilian228 GPTQ-Strata → NVFP4 + Q8_0 mix (cross) | L34 | 95.0 %, 105 GiB | PASS value; **FAIL** rendering: excluded from the Strata set as "restated" (F5) |
| 13 | turboderp → 2.05/3.05/4.05/5.05/6.05 bpw (own ×5, image) | L27–28 | KLD 0.0684/0.0177/0.0067/0.0040/0.0031; PPL 1.4879/1.4075/1.3900/1.3848/1.3825 | PASS (10) |
| 14 | turboderp → UD-IQ1_M … UD-Q4_K_XL (cross ×5, image) | L27–28 | 0.0804/0.0533/0.0349/0.0165/0.0084; 1.5031/1.4619/1.4342/1.4017/1.3926 | PASS values (10); **FAIL** attribution to unsloth (F10) |
| 15 | turboderp → NVFP4 W4A16 / W4A4 (cross ×2, image) | L27–28 | 0.0100/0.0241; 1.3970/1.4199 | PASS (4) |
| 16 | turboderp → baseline / noise floor | L28 / L27 | BF16 PPL 1.38423; noise floor 0.00249 | PASS (2) |
| 17 | unsloth docs → UD-IQ1_S, UD-Q4_K_XL, Q8_0 (own ×3) | docs L96, L102, L105 | 72.5/111.3/188.2 GB; 77.325/92.255/94.122; 0.396070/0.046893/0.026574; 7.2126/1.5468/0.8118 | PASS (12) |
| 18 | AesSedai → Q5_K_M, Q4_K_M, IQ4_XS, IQ3_S, IQ2_S (own ×5) | L15–19 | 153.07/131.98/114.99/105.98/103.63 GiB; 7.32/6.31/5.50/5.07/4.96 bpw; PPL 4.596526/4.613527/4.739310/4.863539/5.043883; +0.1270/+0.4973/+3.2373/+5.9434/+9.8718 %; KLD 0.020218/0.034206/0.076158/0.160378/0.205353 | PASS (25) |
| 19 | AtomicChat → AD-3.84/4.27/5.00 (own ×3) | L83–85 | 84.9/94.5/110.5 GB; 0.2277/0.0842/0.0837; 82.68/89.49/89.55 %; 1.102/1.026/1.026 | PASS (12) |
| 20 | AtomicChat → baseline | L97 | BF16 PPL 4.0445 ± 0.0216 | PASS (1) |
| 21 | nvidia → NVFP4 (own) | L89–111 | 91.5/35.4/90.1/78.3/18.8/74.1/81.0/27.6/82.9 | PASS (9) |
| 22 | nvidia → FP8 baseline | L67–91 | 92.0/34.7/90.8/77.1/16.3/71.9/80.5/28.1/83.3 | PASS (9) |
| 23 | ISTA-DASLab → Q2_0, IQ2_XS, IQ3_XXS, IQ3_S (own ×4) | L124–127 | 66.4/68.0/75.8/83.6 GB; AIME 96.67/96.67/100/100; GPQA 89.39/87.37/91.41/92.93; LCB 81.14/83.43/86.29/86.86; avg 89.07/89.16/92.57/93.26 | PASS (20) |
| 24 | ISTA-DASLab → BF16 baseline | L123 | 354 GB; 100.00/91.92/87.43/93.12 | PASS (5) |
| 25 | ji-farthing → v3 IQ4_KT, v3 IQ3_KT (own ×2) | L110–111, L37–38, L116 | 82.75/72.52 GiB; 0.057803/0.085269; 91.183/89.384 %; 0.055726/0.077747; 95.142/94.653 %; PPL 3.3893/3.4484, 1.7303/1.7522 | PASS (14) |
| 26 | ji-farthing → IQ4_KT_v2 (ladder) | L112, L39, L116 | 0.132805, 86.276 %, 0.124558, 93.216 %, 3.5335, 1.8031; size 80.72 | metrics PASS (6); **FAIL** size: card says 80.72 **GiB**, stored/plotted as 80.72 GB (F2) |
| 27 | ji-farthing_v2 → IQ4_KT_v2 (own) | L73, L40 | 80.72 GiB, 0.132805, 86.276 %, 0.124558, 93.216 % | PASS (5) |
| 28 | ji-farthing_v2 → Previous IQ3_KT / IQ4_KT (ladder ×2) | L71–72 | 0.151454/85.054 %/0.134990/92.131 %; 0.132548/85.738 %/0.124749/92.414 %; sizes 80.22, 88.42 | metrics PASS (8); **FAIL** sizes: GiB on card, GB on site (F2) |
| 29 | TheDrainFlorist 2.1bpw → VQ-2.1bpw (own) | L173 | 44.5 GiB, 652.4 mnats, 72.3 %, 3.559 | PASS (4) |
| 30 | TheDrainFlorist 4.4bpw → VQ-4.4bpw (own); affine q3/q4/q5/q6/q8, VQ-3.2/5.5 (ladder ×7) | L180–188 | 93.6/75/68.7/96/116/111.6/137/178 GiB; 47.7/1312.4/147.3/298.2/97.5/30.9/39.9/22.4 mnats; 92.2/49.5/86.7/77.8/89.0/93.7/93.4/94.8 %; 2.323/14.054/2.300/3.912/2.549/2.244/2.276/2.367 | PASS (32) (unit GiB supplied by a build-time special case, not by the data) |
| 31 | Tylogi → V1-S … V4-XS (own ×6) | L80–85 | 72.536/74.529/78.859/81.952/89.976/93.672 GB; 0.22157/0.19689/0.14630/0.12436/0.08698/0.07524; 83.77/84.72/86.54/87.62/89.55/90.36 % | PASS (18) |
| 32 | Tylogi → unsloth UD-IQ1_S (cross) | L86 | 72.5, 0.39607, 77.33 % | PASS; correctly marked restated (column "Unsloth UD" links unsloth docs) |
| 33 | beamster 2bpw → Sushi-2bpw (own); Sushi-4bpw/3bpw/2.6bpw (ladder) | L57, L49, L52–53 | 34.97/63.68/49.33/43.95 GiB; 0.2080/0.0592/0.1036/0.1355; 85.94/92.89/90.31/89.08 % | PASS (12) |
| 34 | showkieo 3bpw → Sushi-3bpw (own); Sushi-4bpw, affine q3 (ladder) | L118, L116, L120 | 49.33/63.68/54.94 GiB; 0.1047/0.0632/0.1444; 90.34/92.99/88.05 % | PASS values (9); **FAIL** measurer: card is beamster's text (F9) |
| 35 | pipenetwork 6bit → 6bit (own); 8bit, mixed-4_8bit, 4bit (ladder) | L80–83, L57 | 148.0/192.2/106.2/103.8 GB; PPL 4.4767/4.4749/4.5286/5.3914; ΔNLL +0.0013/+0.0009/+0.0128/+0.1872; 81/73/128/145 of 145 | PASS (16); ref_class "other" although reference is bfloat16 (F3) |
| 36 | 6block → IQ4_XS, IQ3_XXS, IQ2_XS-layered, IQ1_M-layered (own ×4) | L40–43 | 118.8/107.4/83.5/76.4 GiB; 5.67/5.13/3.98/3.65 bpw; 1.8378/2.0528/2.7833/4.4105 | PASS (12) |
| 37 | spiritfather i1 → IQ2_M, IQ3_XXS, IQ3_M, IQ4_XS (own ×4) | L48–51 | 100.9/105.6/109.1/116.7 GiB; 5.1252/4.9843/4.9869/4.8502; +5.97/+3.05/+3.11/+0.28 %; 0.150323/0.093681/0.072916/0.039888 | PASS (16) |
| 38 | scorpoon → 3.05/3.15/2.05 bpw h8 SC (own ×3) | L60–62, L31–33 | 49.56/50.44/34.25 GiB; 0.01543/0.01454/0.05761; 1.4041/1.4016/1.4697; +1.41/+1.23/+6.15 % | PASS (12); badge "other or no reference" although reference is "Qwen3.8-Flash-Next BF16" (F3) |
| 39 | scorpoon → turboderp 3.05 h5, 2.05 h4 (cross ×2) | L63–64 | 0.01730/1.4071/+1.63 %; 0.06840/1.4879/+7.46 % | PASS (6) |
| 40 | fuutott → this checkpoint (own) | L107, L63, L126–139 | 0.489, 0.187, 76.1 %, 4.35 (×1.32); 249/90/180/68/587; 152/390; 97,059,902,320 bytes | PASS (13); size basis chosen = disk although L107 gives 35.6 GiB resident → dropped from its own set (F6) |
| 41 | fuutott → Affine Q2; T5 Q2 down; T5 no-imatrix (ladder ×3) | L108–110 | 35.6/30.9/30.9 GiB; 0.477/0.573/0.690; 0.186/0.237/0.327; 76.4/74.0/71.3 %; 4.31/4.67/5.27 | PASS (15) |
| 42 | fuutott → unsloth UD-Q4_K_XL (cross) | L111, L126–139 | 0.036, 0.007, 93.9 %, 3.31; 269/92/192/81/634; 157/429 | PASS (11) |
| 43 | SiliconSpecies → 5bpw, 5bpw-hq, 6bpw-hq (own ×3) | L34–36, L24–26 | 78/78/92 GiB; 0.0357/0.0325/0.0284; 0.0105/0.0095/0.0099; 0.0163/0.0153/0.0124; 92.9/93.1/93.5 % | PASS (15) |
| 44 | SiliconSpecies → beamster Sushi-4bpw; "port" (cross ×2) | L33, L32 | 0.0550/0.0180/0.0248/91.0 %; 0.1416/0.0528/0.0841/86.1 % | PASS (8) |
| 45 | mudler → APEX-I-Nano, Mini (own ×2) | L92, L91, L39–40 | 73.0/78.7 GB; 4.6659/4.4354; +8.2/+2.9 % | PASS (6) |
| 46 | Cyronius → keep1-Q3KXL (own); table deleted (ladder); unsloth UD-Q3_K_XL (cross) | L50, L51, L49, L36 | 64.8/61.2/90 GB; tools 1.000/.975/.900; GSM8K-15 1.000 ×3; MMLU-30 .833/.767/.833; PPL 4.66/4.80/2.40 | PASS (15) |
| 47 | whm0627 → Q2_K_XL-PLElast, IQ1_S-PLElast (own ×2) | L45–49, L27–28 | 79/68 GB; 37/40, 38/40; 20/20, 19/20; PPL 1.084/1.156 | PASS (8) |
| 48 | PaoAI → BALANCED-2.1 (own); BALANCED-2 previous (own) | L163, L60, L74, L139, L59, L76 | 78.27/80.78 GB; 3.54/3.65 bpw; 4.6618/4.6695; −0.16 %; 100/100; 75.7; 86.7 | PASS (10) |
| 49 | ddalcu 4.7bpw → iQ-MLX-4.7bpw (own); mixed-4-8bit (ladder) | L62, L61, L68, L37 | 75.3/75.3 GB; top-1 88.6/89.0/96.0/90.0/81.2 and 87.9/89.0/88.0/87.5/87.0; median 0.0052/0.0066; mean 0.126/0.158; 4.68 bpw | PASS (17) |
| 50 | Maximilian228 Mix-Strata → NVFP4 + Q8_0 mix (own) | L41, L120, L130 | 95.0 %, 97.4 %, 0.0215, 0.0058, 100.0 %, 171.6 GB | PASS (6); the quant is absent from its own auto set (F6) |
| 51 | Maximilian228 Q8_0-Strata → mix, GPTQ, NVIDIA (ladder ×3) | L38, L40–41 | 95.0/93.6/92.9 %; 105/69/69 GiB | PASS (6); duplicate of the curated Strata table (F5) |
| 52 | yamz-labs → EXL3 pack (own); Strata UD-IQ4_XS; Gufo UD-Q4_K_XL (cross ×2) | L83–85, L21, L56 | 95/93.7/111.3 GB; 93.81/93.64/95.36 %; 0.0216/0.0316/0.0156; 0.0008/0.0074/0.0000 | PASS (12); unsloth attribution inferred (F10) |
| 53 | mensaprodigy REAP-384-mlx-mtp → sh0wie lineage (cross ×4) | L131–134 | 93.9/92.1/91.5/88.4 % | PASS values (4); **FAIL** "measured by mensaprodigy" — card L127 says reported by Sh0wie (F1) |
| 54 | agentionai ROCmFP4 → AesSedai IQ4_XS (cross) | L171 | 117.13 GiB, +3.12 % | PASS values (2); **FAIL** head-to-head labels it a ctx-2,048 measurement by agentionai; card L163–165 says quoted vs AesSedai's own reference (F1) |
| 55 | batiai → IQ3_XXS, IQ4_XS, Q4_K_M, Q6_K (own ×4) | L72–75, L229, L133, L139, L220 | 79.2/90.8/111.0/141.2 GiB; 70.0 %; 7/7, 70.4 %, 73.1 %; 70.5 % | PASS (9) |
| 56 | Ar3ntiz → SUC1 side8 (own); GGUF Q4 (ds4), MLX affine 4-bit (cross) | L49, L27, L55–57, L63–64 | 96.32 GB, 4.35 bpw; 0.064/0.913/4.704; 97/17/15/6/20/19/20; 177.28 GB, 0.061, 99/18/16/7/19/19/20; 0.202 | PASS (23) |
| 57 | sleepyeldrazi → IQ3_XXS-HCQ5_0 (own); ISTA IQ3_XXS (cross) | L68, L57, L62 | 46.2 GiB, 47 tokens; 3.9044 | PASS values (3); **FAIL** cross chip "measured by sleepyeldrazi" for an ISTA figure the card quotes (F1) |
| 58 | groxaxo unsloth-shaped → EXL3 3.82 (own) | L175–180, L51–52 | 9/10, 4/5, 5/5, 5/5, 2/2, 23/25; ~104 GB; 3.82 | PASS (8) |
| 59 | inferencerlabs Q4i → Q3 … Q9 (ladder ×7) | L34–40 | 168.0/1.33593/1.21093/1.23437/1.21875/1.21875/1.20312; 43.45/91.65/97.70/95.05/96.65/97.65/97.80 %; 72.57/17.28/10.65/17.28/12.03/9.92/9.60 % | PASS values (21); **FAIL** model: L27 says these are GLM 5.1 figures (F8) |
| 60 | edougawa PLE-MMap → NVFP4 body + mmap PLE (own) | L84, L88, L47 | 0.995540; 0.995620; 0.995146; 69.9 GiB | PASS (4) |
| 61 | otheru → abliterated IQ4_XS, native .hgn, mtp.hgn (own ×3) | L41–43, L154–155 | 97,615,265,184 / 124,429,843,520 / 1,523,566,720 bytes; 7.6200 %; 2.6551 % | PASS (5) |
| 62 | kingjones777 FAST-imatrix → ROCmFP4-FAST imatrix (own); plain FAST (cross) | L57–58, L60, L71 | 5.0337 ± 0.031, −5.9 %, 5.1075, 87.9 GiB, 4.27 bpw; 5.3465 ± 0.034 | PASS (6) (Δ is vs the sibling build, F17) |
| 63 | kingjones777 Uncensored FAST → own; aligned build (ladder) | L211–213, L130 | 6/8, 11, 22; 6/8, 10, 0; 87.94 GiB, 4.27 bpw | PASS (8) |
| 64 | myllmbox hibrid47 → own; hibrid46 (ladder) | L30, L31 | 26 good / 3 partial / 3 broken; "about half/half" | PASS (3 + 1 non-numeric) |
| 65 | apetersson Abliterated → Q5_K_M (own); stock Q5 (ladder); shard 2 (own) | L38–41, L160–163, L55 | 7/8, 7/8, 10/12, 6/8; 0.97–1.00, 0.99–1.01, 0.86, 0.34; 5/8, 6/8, 10/12, 6/8; 95.37 GiB | PASS (13) |
| 66 | octolix → Octojet (own); local-inference-lab NVFP4 on TF 0.6.0 (cross) | L105–106 | 245/155; 246/158 | PASS (4) |
| 67 | Sawfwair Activation-3bit (+ Native-PLE) → own; Q4 (ladder) | L67, L69, L22; L77–82 | 58 of 61; all 16; 83.49 GiB; 77,180,432,496 bytes; 19.58 % | PASS (7) |
| 68 | Wladastic Ombro-v9 → UD-Q4_K_XL / UD-Q5_K_XL / UD-IQ3_XXS sizes (cross ×3) | L27, L31, L90 | 111.3 / 158.3 / 82.0 GB | PASS values (3); "quoted, not measured" flag dropped (F1) |

Totals: 68 rows, ≈470 individual numbers; 0 numeric transcription failures; 9 rows FAIL on attribution/unit/model (rows 12, 14, 26, 28, 34, 53, 54, 57, 59).

---

## Gate-gap summary (for test_gates.py)

| Mutation (scratch copy) | verify.py | verify_columns.py | build.py | Caught? |
|---|---|---|---|---|
| Pivoted-table column swap (perkel Tiny↔Small, same row) | pass | pass | pass | **no** |
| Sign flip in published string (PaoAI −0.16 % → +0.16 %) | pass | pass | pass | **no** |
| Unit flip GiB→GB (AesSedai Q5_K_M) | pass | pass | pass | **no** |
| Variant-name swap keeping `L` (AesSedai Q5_K_M↔Q4_K_M) | pass | pass | pass | **no** |
| Missing unit on a GiB size (ji-farthing, live data) | pass | pass | pass | **no** (F2) |
| Quoted row filed as cross measurement (mensaprodigy, live data) | pass | pass | pass | **no** (F1) |
| Stated-BF16 reference classed "other" (scorpoon, live data) | n/a | n/a | pass | **no** (F3) |

## Recommended order of work
1. F1 (quoted rows + entry flags) and F8 — data honesty about who measured what.
2. F2 units; F3 ref_class; F4 method text.
3. F5/F6/F7 set construction; F9/F10/F12 attribution and copy flags; F13 frontier; F14 note; F15 stat.
4. F11 gates added to `test_gates.py` so 1–3 cannot regress.
5. Minors.
