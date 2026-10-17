# Quant Claims Ledger — review pack

Prepared 2026-10-17 for an independent reviewer. You do not need the conversation that produced this; everything needed to re-check the work offline is in this archive.

## 1. What this is

A public comparison of every Hugging Face quantization of **Qwen/Qwen3.8-Flash-Next**, built only from each publisher's **own claims** (model cards and, for Unsloth, its docs page). Nothing was re-measured. The product is a static site (`site/index.html`, for GitHub Pages) plus a JSON dataset (`site/data.json`).

**Goal:** let a reader see what each quant's publisher claims, how it was measured, how claims relate where they are comparable, and where they disagree, without the site implying rankings the evidence does not support.

**Scope at capture (2026-10-17 for the first 44 cards; the rest captured the same day):** 453 repositories tagged `base_model:quantized:Qwen/Qwen3.8-Flash-Next`. Formats: {'GGUF': 105, 'NVFP4': 21, 'AutoRound': 1, 'MLX': 112, 'FP8': 5, 'MXFP4': 3, 'AWQ': 3, 'EXL3': 14, 'NVFP4+FP8': 2, 'MXFP4+FP8': 2, 'compressed-tensors W4A16': 3, 'EXL3 (sushi pack)': 6, 'GLQ': 1, 'MLX (JANGH)': 2, 'Paiton (W3A8)': 1, 'MegaCapybara (repacked EXL3)': 1, 'HGN (Halogen)': 3, 'CNQ (NVFP4)': 1, 'GGUF (CIRU compact expert format)': 1, 'GPTQ': 1, 'NVFP4 (+MXFP8)': 1, 'GGUF (PQ2_0 ternary experts)': 1, 'GGUF (ds4)': 1, 'AutoRound (GPTQ-format)': 1, 'GGUF (ROCmFP4)': 4, 'MXFP4/FP8': 1, 'NVFP4 (+FP8 PLE)': 1, 'GGUF (custom qwen4exp schema, ds4)': 2, 'MXFP8 (compressed-tensors)': 1, 'AMX_Q8R (.amxq)': 1, 'strixw': 1, 'BF16 safetensors (JiRack port); GGUF planned': 1, 'NVFP4+Q8_0 (Strata pack)': 1, 'MLX (oMLX oQ4e)': 2, 'MLX (oMLX oQ5e)': 2, 'MLX (AFM)': 2, 'compressed-tensors int6 (W6A16)': 1, 'MFQ (.mfq)': 1, 'GGUF (TRUSS-only)': 1, 'Strata pack (Q8_0 experts)': 1, 'MLX (TurboQuant)': 1, 'AutoRound (compressed-tensors)': 1, 'compressed-tensors (MXFP4/MXFP8)': 3, 'compressed-tensors (W4A16)': 1, 'MLX (oMLX fork, ternary)': 1, 'MLX (oMLX)': 2, 'GGUF (EXL3 expert tensors)': 1, 'radiance .rad (EXL3 experts)': 1, 'NVFP4 (W4A4) + paired-4:8 sparsity': 1, 'Paiton W4A8 (INT4/INT3 hybrid experts)': 1, 'uint3 g64 (ModelOpt mixed)': 1, 'NInfer container (from GGUF)': 1, 'MXFP4-Mixed (llm_compressor)': 1, 'MLX (mlx-ternary mixed)': 1, 'halogen .hgn': 1, 'BFP4 (Tenstorrent)': 1, 'trellis (EXL3/QTIP family, .ninfer)': 1, 'GGUF (MTP sidecar only)': 1, 'GGUF (custom qwen4exp schema, ds4 SSD-PLE)': 1, 'MLX (gbx_lm paged containers)': 1, 'GGUF (MTP draft head only)': 6, 'GGUF control vector': 1, 'NVFP4 (ModelOpt safetensors)': 1, 'NVFP4 (compressed-tensors nvfp4-pack-quantized, W4A16)': 1, 'lily qwen4_exp-affine-v1 (safetensors, engine-specific)': 2, 'NInfer mixed (NVFP4/FP8/INT4)': 1, 'GGUF (ROCmFP4, Q4_0_ROCMFP4_STRIX)': 1, 'NVFP4 (ModelOpt mixed: NVFP4 W4A16 experts + FP8 block attention)': 1, 'NVFP4 experts + MLX affine 4-bit (Octojet-only)': 1, 'GGUF + halogen .hgn': 1, 'GGUF (MTP draft head only; ROCmFP4 output tensor)': 1, 'MoEspresso package (IQ_K/K-quant MLX)': 1, 'DACAN pack (GGUF dense + expert .bin)': 1, 'EXL3 (GPTQModel fork)': 1, 'MLX (mlx-serve layout)': 1, 'not stated': 4, 'SUC1': 1, 'GGUF (PLE sidecar tables only)': 1, 'MLX (MTPLX layout)': 3, 'NVFP4 (ModelOpt, W4A16)': 2, 'NVFP4/MXFP8': 1, 'EXL3 (Sushi pack)': 1, 'MLX (MTPLX pack)': 1, 'GGUF (ROCmFP4, MTP draft head only)': 1, 'NVFP4 experts + INT3 PLE (compressed-tensors style)': 1, 'MLX (mlx-serve pack)': 4, 'FinchMoE .finch (custom streaming layout)': 1, 'compressed-tensors (MXFP4 experts + MXFP8 PLE)': 1, 'NVFP4 (ModelOpt)': 5, 'FP8 (compressed-tensors, FP8_DYNAMIC)': 1, 'MLX (B.E.A.M. layout)': 1, 'compressed-tensors AWQ INT4 (W4A16)': 1, 'compressed-tensors W4A16 (GPTQ) + INT4 PLE + FP8 KV scales': 1, 'MLX (per repo name)': 2, 'FP8 (per repo name)': 1, 'GGUF (ROCmFPX extension types)': 1, 'MLX (oMLX oQ)': 1, 'NVFP4 (ModelOpt) + NVFP4 PLE sidecar': 1, 'compressed-tensors (INT8/INT4 pack-quantized)': 1, 'MLX (oQ)': 1, 'MLX (MTPLX)': 2, 'MLX-style (lisa runtime)': 1, 'AWQ (moe_wna16) + W8A16 PLE': 1, 'MLX (mlx-serve)': 3, 'Glyd proprietary format': 1, 'W4A16 (INT4 AutoRound experts) + INT8 GDN + FP8 PLE': 1, 'CMF (cortiq)': 1, 'compressed-tensors mixed-precision (MXFP4/MXFP8)': 1, 'NVFP4 (ModelOpt 0.46.0)': 1, 'GGUF (Mica)': 1, 'NVFP4A16 (compressed-tensors)': 1, 'MLX (mlx-optiq)': 1, 'GGUF (MTP block only)': 1, 'NVFP4 W4A16 (ModelOpt)': 1, 'GGUF (per-layer library)': 1, 'GGUF (MTP draft head)': 3, 'GGUF (ROCmFP4, fork-only)': 2, 'GGUF control vectors': 1, 'NVFP4 (compressed-tensors)': 1, 'compressed-tensors (INT8/INT4 experts, BF16 attention)': 1, 'NVFP4/FP8 (per repo name)': 1, 'NVFP4 + FP8 (ModelOpt mixed)': 1, 'GGUF (ROCmFPX types)': 1, 'NVFP4 (ModelOpt FP4)': 1, 'GGUF (Hadamard-rotated, patched runtime)': 1, 'NInfer container (NVFP4/FP8 + INT4 PLE)': 1, 'compressed-tensors (MXFP4)': 1, 'GGUF (custom CIRU IU4 format)': 1, 'GGUF (MTP draft heads)': 1, 'ModelSlim INT4 W4 g128 (Ascend 310P)': 1, 'NVFP4 mixed (vLLM)': 1, 'GGUF (private Q3_PLE type, patched llama.cpp)': 1, 'MLX (AXQuant)': 1, 'GGUF (bundle)': 1, 'compressed-tensors AWQ INT4 + FP8 PLE': 1, 'MLX (mlx-serve mixed)': 1, 'EXTENSOR GGUF (not llama.cpp compatible)': 1, 'AutoRound W4A16 (auto_gptq) + FP8 PLE': 1, 'AutoRound W4A16 (GPTQ packing) + FP8 PLE': 1, 'NVFP4 (ModelOpt W4A16_NVFP4)': 1, 'Quark mixed INT4/INT8 (AWQ)': 1, 'Orinfer safetensors (E8P)': 1, 'Ascend ModelSlim W8A8_DYNAMIC (ascend_v1)': 1, 'NVFP4 (ModelOpt, per HF tags)': 1, 'compressed-tensors NVFP4/FP8 (per HF tags)': 2, 'halogen .hgn (custom)': 1, 'CQ 4-bit (custom)': 1, 'safetensors (2-bit, per HF tags)': 1}.

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

### verify.py (verbatim check, all repos)
exit code 0
```
repos covered: 453/453
values verified verbatim: 3506
image-sourced values (see verify_images.py): 37
signed values checked for sign: 97 (sign failures: 0)
sizes checked for unit next to the number: 789 (unit failures: 0)
cross/ladder entries with quoted-context wording: 8
failures: 0
```

### verify_columns.py (table-cell / metric binding)
exit code 0
```
table-cell bindings checked: 1391; unexplained mismatches: 0
row-label bindings checked: 974 (pivoted-column checks: 245; single-subject tables naming no entry, not checkable: 316); unexplained mismatches: 0
```

### verify_images.py (chart-image transcriptions)
exit code 0
```
image claims checked: 14, failures: 0
```

### detect_copies.py (copied cards, share >= 0.30 listed; >= 0.60 treated as copied)
exit code 0
```
1.00  FenomAI/Qwen3.8-Flash-Next-GSQ-RCO-GGUF                                <- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (48/48)
1.00  Guile/Qwen3.8-Flash-Next-GGUF                                          <- bartowski/Qwen3.8-Flash-Next-GGUF (6/6)
1.00  Kashapps/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF               <- kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF (21/21)
1.00  Manav1011/Qwen3.8-Flash-Next-NVFP4                                     <- RadixArk/Qwen3.8-Flash-Next-NVFP4 (11/11)
1.00  ScottzillaSystems/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4               <- dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4 (82/82)
1.00  TracNetwork/mayhem-catalog-Qwen-Qwen3-8-Flash-Next-NVFP4               <- RadixArk/Qwen3.8-Flash-Next-NVFP4 (11/11)
1.00  aixiaoma/Qwen3.8-Flash-Next-W4A16                                      <- VnimanieAI/Qwen3.8-Flash-Next-W4A16 (15/15)
1.00  cc1244578151/Qwen3.8-Flash-Next-GSQ-RCO-GGUF                           <- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (48/48)
1.00  cc1244578151/myqwen3.8flashgsqrco                                      <- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (48/48)
1.00  kstoyanov99/qwen38-flash-next-nvfp4                                    <- RadixArk/Qwen3.8-Flash-Next-NVFP4 (11/11)
1.00  lhoestq/Qwen3.8-Flash-Next-NVFP4-Michel                                <- nvidia/Qwen3.8-Flash-Next-NVFP4 (6/6)
1.00  maclocal/Qwen3.8-Flash-Next-AFM-MLX-4bit                               <- scouzi1966/Qwen3.8-Flash-Next-AFM-MLX-4bit (9/9)
1.00  qtum/Qwen3.8-Flash-Next-GGUF                                           <- 6block/Qwen3.8-Flash-Next-GGUF (9/9)
1.00  sorenmat/Qwen3.8-Flash-Next-NVFP4                                      <- nvidia/Qwen3.8-Flash-Next-NVFP4 (6/6)
1.00  thoddnn/Qwen3.8-Flash-Next-4bit                                        <- mlx-community/Qwen3.8-Flash-Next-4bit (6/6)
0.97  Robot-Haus/Qwen3.8-Flash-Next-oQ4e-fp16-mtp                            <- evsinlb/Qwen3.8-Flash-Next-oQ4e-fp16-mtp (29/30)
0.92  vaultai/Qwen3.8-Flash-Next-MLX-oQ4-MTP                                 <- TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP (11/12)
0.90  mk2878d/Qwen3.8-Flash-Next-GGUF                                        <- AtomicChat/Qwen3.8-Flash-Next-GGUF (46/51)
0.85  lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000             <- RadixArk/Qwen3.8-Flash-Next-NVFP4 (11/13)
0.77  eemin/halogen-qwen3.8-flash-next-copy                                  <- peonist-ai/halogen-qwen3.8-flash-next (17/22)
0.71  auto7x/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4-GGUF    <- jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4 (5/7)
0.71  ddalcu/Qwen3.8-Flash-Next-MLX-Serve-8bit                               <- airagrp/Qwen3.8-Flash-Next-mixed-6-8-ple8-mlx-serve (5/7)
0.50  SargeDev/qwen3.8-flash-next-lil-rascal-uncensored                      <- Baekpica/Qwen3.8-Flash-Next-GGUF (2/4)
0.50  d9beuD/Qwen3.8-Flash-Next-oQ8e-mtp                                     <- mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp (2/4)
0.39  lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8                                <- RadixArk/Qwen3.8-Flash-Next-NVFP4 (11/28)
0.33  Robot-Haus/Qwen3.8-Flash-Next-MLX-oQ3-MTP                              <- TensorFold/Qwen3.8-Flash-Next-MLX-oQ3-MTP (8/24)
0.33  d9beuD/Qwen3.8-Flash-Next-oQ6e-mtp                                     <- mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp (2/6)
0.33  spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF                   <- argyelan/Qwen3.8-Flash-Next-GGUF (2/6)
0.31  airagrp/Qwen3.8-Flash-Next-mixed-6-8-ple8-mlx-serve                    <- ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit (4/13)
```

### diff_blind.py — first 44 repos vs blind auditor
exit code 0
```
ours 674 values, auditor 1190, agreed 624; repos with differences: 36
```

### diff_blind.py — 152 later measured repos vs two blind auditors
exit code 0
```
ours 2002 values, auditor 4229, agreed 1957; repos with differences: 132
```

### build.py (plausibility assertions + site build)
exit code 0
```
{"repos": 453, "variants": 727, "values": 2980, "with_quality": 196, "bf16_ref": 91, "unreviewed": 0, "no_quality": 257, "cross": 152, "quoted": 23, "other_model": 7} | rows 1103
```

### test_gates.py (mutation tests: gates must fail closed)
exit code 0
```
[PASS] absent value blocks build and pack
[PASS] swapped KLD/PPL columns caught
[PASS] image baseline checked
[PASS] unicode minus keeps sign
[PASS] no invisible metrics in ledger
[PASS] linked estimates disabled
[PASS] auto sets: one basis, no quoted cross rows
[PASS] unique explorer ids
[PASS] pivoted-table column swap caught (perkel Tiny/Small)
[PASS] sign flip caught (PaoAI)
[PASS] GiB->GB unit flip caught (AesSedai)
[PASS] variant-name swap keeping L caught (AesSedai)
[PASS] quoted-context entry without measured= caught (mensaprodigy)
[PASS] size without unit caught (AesSedai)
```

### test_claims.py (regression checks for the cold review, F1-F24)
exit code 0
```
[PASS] F1 no flagged-quoted row is kind=cross
[PASS] F1 no row named 'quoted'/'-reported' is kind=cross
[PASS] F1 quoted rows exist and are labelled 'quoted by'
[PASS] F1 AesSedai/agentionai pseudo-disagreement removed from head-to-head
[PASS] F1 set turboderp has no quoted points
[PASS] F1 set gyro has no quoted points
[PASS] F1 set ap has no quoted points
[PASS] F1 set strata has no quoted points
[PASS] F1 set unsloth has no quoted points
[PASS] F1 set aessedai has no quoted points
[PASS] F1 set drainflorist has no quoted points
[PASS] F1 set jifarthing has no quoted points
[PASS] F1 set atomicchat has no quoted points
[PASS] F1 set nvidia has no quoted points
[PASS] F1 set ista has no quoted points
[PASS] F1 set a-tylogi-ew-mfq has no quoted points
[PASS] F1 set a-beamster-sushi-2bpw has no quoted points
[PASS] F1 set a-pipenetwork-mlx-6bit has no quoted points
[PASS] F1 set a-6block-gguf has no quoted points
[PASS] F1 set a-spiritfather-cyberneurova-i1-gguf has no quoted points
[PASS] F1 set a-ji-farthing-ik-llama-gguf-v2 has no quoted points
[PASS] F1 set a-cyronius-131b-a6b-gguf has no quoted points
[PASS] F1 set a-scorpoon-exl3-sc has no quoted points
[PASS] F1 set a-bitlamas-gguf has no quoted points
[PASS] F1 set a-siliconspecies-exl3-sushi has no quoted points
[PASS] F1 set a-final-bench-pocket-gguf has no quoted points
[PASS] F1 set a-mudler-apex-gguf has no quoted points
[PASS] F1 set a-perkel-mc has no quoted points
[PASS] F1 set a-jcbtc-qwen3-8-flash-ciru-strix-orca has no quoted points
[PASS] F1 set a-kingjones777-uncensored-rocmfp4-fast-imatrix-ggu has no quoted points
[PASS] F1 set a-whm0627-177b-a3b-fits64gb-plelast-gguf has no quoted points
[PASS] F1 set a-paoai-paoai-strix-balanced-2-gguf has no quoted points
[PASS] F1 set a-drluoto-agenticrequant-q5k-gguf has no quoted points
[PASS] F1 set a-ar3ntiz-qwen3-8-flash-next-side8 has no quoted points
[PASS] F1 set a-ddalcu-mlx-serve-iq-mlx-4-7bpw has no quoted points
[PASS] F1 set a-jcbtc-qwen3-8-flash-ciru-strix-iu4 has no quoted points
[PASS] F1 set all has no quoted points
[PASS] F2 every size carries its unit
[PASS] F2 ji-farthing IQ4_KT_v2 plotted as 86.67 GB (80.72 GiB)
[PASS] F3 stated-BF16 references classed bf16
[PASS] F3 negated / component / Q8_0 references not classed bf16
[PASS] F4 no claim that estimates are shown
[PASS] F4 no claim that every card was blind re-extracted
[PASS] F5 strata set includes the NVFP4 + Q8_0 mix row
[PASS] F5 no duplicate Maximilian228 auto sets
[PASS] F6 every auto set contains the title repo's own quant
[PASS] F7 beamster set includes the rival packs beamster scored
[PASS] F8 no GLM 5.1 rows in the ledger
[PASS] F9 showkieo has no comparable set
[PASS] F9 showkieo marked as copy of beamster
[PASS] F9 beamster/showkieo not described as disagreeing cards
[PASS] F10 turboderp's UD rows marked 'repo not named on card'
[PASS] F10 inference is visible in the ledger
[PASS] F12 Kashapps copy source is kingjones777 Uncensored-FAST
[PASS] F13 Q8_0 (reference) flagged is_ref
[PASS] F13 nvidia FP8 baseline flagged is_ref
[PASS] F14 unsloth note does not speculate
[PASS] F19 head-to-head does not state an AesSedai method the card lacks
[PASS] F15 header stat equals measured, uncopied rival rows (152)
[PASS] F16 no raw basis keys in the page
[PASS] F17 PPL change column names its baseline
[PASS] F18 TheDrainFlorist shared table uses one size basis
[PASS] F21 'kl' metrics rendered in the main KLD column
[PASS] F24 ISTA baseline size cites its card line
```

### test_ui.py (Playwright: explorer, filters, layout at 1366px and 390px)
exit code 0
```
[PASS] desk: no horizontal page scroll (got 0px)
[PASS] desk: unique DOM ids []
[PASS] desk: set options unique (33)
[PASS] desk: chart fits its box
[PASS] desk: picker filter narrows chart (12->5, list 5)
[PASS] desk: untick matches clears them
[PASS] desk: reset restores
[PASS] desk: format chip filters (5)
[PASS] desk: legend hides a repo
[PASS] desk: point click pins tooltip with source link
[PASS] desk: pinned point selects its table row
[PASS] desk: tooltip stays on screen
[PASS] desk: Escape closes tooltip
[PASS] desk: frontier drawn and marked
[PASS] desk: table sorts by size
[PASS] desk: size limit filters table (7 rows, max 4.09 <= 4.09)
[PASS] desk: size limit filters chart too
[PASS] desk: KLD limit filters (5 rows, max 0.01 <= 0.0114)
[PASS] desk: efficient-only rows are all unbeaten
[PASS] desk: efficient set matches independent Pareto computation (6 vs 6)
[PASS] desk: frontier marks == unbeaten quants
[PASS] desk: reference build never listed as efficient (8 rows)
[PASS] desk: frontier controls hidden on the not-comparable set
[PASS] desk: turboderp UD rows labelled and not cross-linked [[True, 0], [True, 0]]
[PASS] desk: size-budget presets offered (6)
[PASS] desk: preset ≤ 48 GB caps size (47.9)
[PASS] desk: size-type selector on the mixed set (8 types)
[PASS] desk: size type 'download' shows exactly its 82 points (82 of 203 shown)
[PASS] desk: linear scale toggle
[PASS] desk: explorer comes before head-to-head
[PASS] desk: shared view link restores set, size cap and efficient-only ['gyro', True]
[PASS] desk: find lists sets containing UD-Q4_K_XL (9)
[PASS] desk: find jumps to set and pins the quant
[PASS] desk: every set renders []
[PASS] desk: quant list has room in the 'all' set (648px)
[PASS] desk: extra format chips collapsed
[PASS] desk: expand makes list full width (1.00)
[PASS] desk: 'all' view filterable by format text
[PASS] desk: ledger multi-word search (31 of 1103 rows)
[PASS] desk: notes search (10 of 215 notes)
[PASS] desk: no-claims status filter (238 of 238 repos)
[PASS] desk: still no horizontal scroll after interaction (0px)
[PASS] desk: no JS errors []
[PASS] phone: no horizontal page scroll (got 0px)
[PASS] phone: unique DOM ids []
[PASS] phone: set options unique (33)
[PASS] phone: chart fits its box
[PASS] phone: picker filter narrows chart (12->5, list 5)
[PASS] phone: untick matches clears them
[PASS] phone: reset restores
[PASS] phone: format chip filters (5)
[PASS] phone: legend hides a repo
[PASS] phone: point click pins tooltip with source link
[PASS] phone: pinned point selects its table row
[PASS] phone: tooltip stays on screen
[PASS] phone: Escape closes tooltip
[PASS] phone: frontier drawn and marked
[PASS] phone: table sorts by size
[PASS] phone: size limit filters table (7 rows, max 4.09 <= 4.09)
[PASS] phone: size limit filters chart too
[PASS] phone: KLD limit filters (5 rows, max 0.01 <= 0.0114)
[PASS] phone: efficient-only rows are all unbeaten
[PASS] phone: efficient set matches independent Pareto computation (6 vs 6)
[PASS] phone: frontier marks == unbeaten quants
[PASS] phone: reference build never listed as efficient (8 rows)
[PASS] phone: frontier controls hidden on the not-comparable set
[PASS] phone: turboderp UD rows labelled and not cross-linked [[True, 0], [True, 0]]
[PASS] phone: size-budget presets offered (6)
[PASS] phone: preset ≤ 48 GB caps size (47.9)
[PASS] phone: size-type selector on the mixed set (8 types)
[PASS] phone: size type 'download' shows exactly its 82 points (82 of 203 shown)
[PASS] phone: linear scale toggle
[PASS] phone: explorer comes before head-to-head
[PASS] phone: shared view link restores set, size cap and efficient-only ['gyro', True]
[PASS] phone: find lists sets containing UD-Q4_K_XL (9)
[PASS] phone: find jumps to set and pins the quant
[PASS] phone: every set renders []
[PASS] phone: quant list has room in the 'all' set (506px)
[PASS] phone: extra format chips collapsed
[PASS] phone: expand makes list full width (1.00)
[PASS] phone: 'all' view filterable by format text
[PASS] phone: ledger multi-word search (31 of 1103 rows)
[PASS] phone: notes search (10 of 215 notes)
[PASS] phone: no-claims status filter (238 of 238 repos)
[PASS] phone: still no horizontal scroll after interaction (0px)
[PASS] phone: no JS errors []
```

## 4. Comparable sets

| id | title | points | metrics | reference | type |
|---|---|---|---|---|---|
| turboderp | turboderp · ExLlamaV3 quality benchmark | 12 | kld, ppl | Full-precision model | curated |
| gyro | agentionai · Gyro card | 9 | kld, top1 | unsloth Q8_0 | curated |
| ap | agentionai · AP card | 8 | kld, top1, kld_code, top1_code | Not named on the card ('the same source') | curated |
| strata | Maximilian228 · Strata | 10 | top1 | Maximilian228's own Q8_0 | curated |
| unsloth | unsloth · docs page | 10 | kld, top1, kld_p999 | Not stated | curated |
| aessedai | AesSedai · model card | 10 | kld, ppl, ppl_delta | Not stated on card (repo logs: BF16) | curated |
| drainflorist | TheDrainFlorist · VQ ladder | 9 | kld, top1, ppl | bf16 teacher (cached top-64 logits) | curated |
| jifarthing | ji-farthing · ik_llama.cpp | 3 | kld, top1, kld_code, top1_code, ppl | BF16 GGUF | curated |
| atomicchat | AtomicChat · model card | 3 | kld, top1, ppl_ratio | BF16 logits | curated |
| nvidia | nvidia · NVFP4 vs official FP8 | 2 | gpqa2, hle, tau2, mmmu, scicode, aalcr, ifbench, omni, tb21 | Qwen's official FP8 release (plotted as a point) | curated |
| ista | ISTA-DASLab · benchmarks | 5 | task_avg, gpqa, lcb, aime | Own BF16 run | curated |
| a-tylogi-ew-mfq | Tylogi · EW-MFQ | 10 | kld, top1 | not stated ('the reference model') | auto |
| a-beamster-sushi-2bpw | beamster · Sushi-2bpw | 9 | kld, top1 | bf16 checkpoint | auto |
| a-pipenetwork-mlx-6bit | pipenetwork · MLX-6bit | 4 | ppl | bfloat16 (upstream), identical windows through the same runtime | auto |
| a-6block-gguf | 6block · GGUF | 4 | ppl | BF16 master (own run) | auto |
| a-spiritfather-cyberneurova-i1-gguf | spiritfather · cyberneurova-i1-GGUF | 4 | kld, ppl, ppl_delta | Q8_0 (own build), not BF16 | auto |
| a-ji-farthing-ik-llama-gguf-v2 | ji-farthing · ik-llama-GGUF_v2 | 3 | kld, top1 | BF16 source (same references for all runs) | auto |
| a-cyronius-131b-a6b-gguf | Cyronius · 131B-A6B-GGUF | 3 | ppl | unsloth UD-Q3_K_XL (the unpruned base quant, not BF16) | auto |
| a-scorpoon-exl3-sc | scorpoon · exl3-SC | 3 | kld, ppl, ppl_delta | Qwen3.8-Flash-Next BF16 | auto |
| a-bitlamas-gguf | bitlamas · GGUF | 3 | top1, ppl | unsloth UD-Q4_K_XL (not BF16) for KLD/top-1; Qwen BF16 for down-projection error | auto |
| a-siliconspecies-exl3-sushi | SiliconSpecies · EXL3-Sushi | 3 | top1 | bf16 teacher (whole model) | auto |
| a-final-bench-pocket-gguf | FINAL-Bench · POCKET-GGUF | 2 | ppl | none (compared to unsloth UD-IQ4_XS, not BF16) | auto |
| a-mudler-apex-gguf | mudler · APEX-GGUF | 2 | ppl, ppl_delta | BF16 | auto |
| a-perkel-mc | perkel · MC | 2 | top1 | Qwen's BF16 weights computed with FP32 math | auto |
| a-jcbtc-qwen3-8-flash-ciru-strix-orca | jcbtc · Qwen3.8-Flash-CIRU-STRIX-Orca | 2 | ppl | retained (original) Orca CIRU GGUF, not BF16 (card: full-precision Orca logit reference not available) | auto |
| a-kingjones777-uncensored-rocmfp4-fast-imatrix-ggu | kingjones777 · Uncensored-ROCmFP4-FAST-imatrix-GGUF | 2 | ppl | none (no BF16 or base-model PPL; compared to own non-imatrix sibling) | auto |
| a-whm0627-177b-a3b-fits64gb-plelast-gguf | whm0627 · 177B-A3B-fits64GB-PLElast-GGUF | 2 | ppl | none (no BF16 or unsloth original compared) | auto |
| a-paoai-paoai-strix-balanced-2-gguf | PaoAI · PaoAI-STRIX-BALANCED-2-GGUF | 2 | ppl | none (no BF16 run; PPL compares the publisher's two files) | auto |
| a-drluoto-agenticrequant-q5k-gguf | drluoto · AgenticRequant-Q5K-GGUF | 2 | ppl | unsloth UD-IQ4_XS (source quant, not BF16) | auto |
| a-ar3ntiz-qwen3-8-flash-next-side8 | Ar3ntiz · qwen3.8_flash_next_side8 | 2 | kld | the original model | auto |
| a-ddalcu-mlx-serve-iq-mlx-4-7bpw | ddalcu · MLX-Serve-iQ-MLX-4.7bpw | 2 | kld | bf16 model's own logits | auto |
| a-jcbtc-qwen3-8-flash-ciru-strix-iu4 | jcbtc · Qwen3.8-Flash-CIRU-STRIX-IU4 | 2 | kld, top1 | BF16 full-vocabulary reference | auto |
| all | Every claim together (not comparable) | 203 | kld, top1 | Mixed | n/a |

## 5. External review dispositions (GPT "Astra")

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


## 6. Internal adversarial page review

An internal adversarial page review was started but interrupted before it reported; it was not rerun. Two external reviews supersede it: GPT "Astra" and a cold review by Fable (`audit/fable_review.md`, verdict "ship after fixes"). Dispositions for both are in §5.

## 7. Reference classes (repos)
{'unstated': 7, 'bf16': 101, 'other': 102, 'none': 237, 'q8': 4, 'unclear': 2}

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
