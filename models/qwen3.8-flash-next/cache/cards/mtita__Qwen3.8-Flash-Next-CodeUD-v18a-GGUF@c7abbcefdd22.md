---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - qwen
  - moe
  - gated-deltanet
  - ssd
  - llama.cpp
  - rocm
  - rdna4
  - amd
  - quantization
  - text-generation
quantized_by: mtita
---

# Qwen3.8-Flash-Next-CodeUD-v18a-GGUF

![GGUF](https://img.shields.io/badge/format-GGUF-4c6ef5)
![bpw](https://img.shields.io/badge/trunk-3.13_bpw-2f9e44)
![VRAM](https://img.shields.io/badge/offload-100%25_GPU_2x32GB-e03131)
![context](https://img.shields.io/badge/context-262k-1c7ed6)
![license](https://img.shields.io/badge/license-Qwen_Community_1.0-9c36b5)
![status](https://img.shields.io/badge/status-educational_build-f59f00)

## 🎯 Purpose

**This is primarily an educational project.** It exists to dissect and learn how
[`Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) works — its hybrid
**MoE + Gated DeltaNet** architecture, its 51.2B-parameter n-gram PLE table, its expert
routing and linear-attention state — and to ship the **best-possible-quality GGUF
quantization** that runs **100% GPU-resident** on a pair of AMD RDNA4 (`gfx1201`) Radeon AI
PRO R9700 accelerators: **32 GB each, 64 GB total VRAM**.

It comes with **no guarantees whatsoever** — not of quality, stability, correctness, or
fitness for any particular purpose. Use at your own risk.

The base model is ~16 bits/weight and needs **>120 GB of VRAM** for weights alone. It does
not fit on any 64 GB workstation. The entire project is a long, recorded experiment in
spending a hard GPU budget on the right tensors — **every build was a test, every test
produced one lesson, and every lesson drove the next build.** The full story is the main
content of this card.

## ⚙️ Hardware geometry (constant throughout)

| Item | Value |
|---|---|
| GPUs | 2 × AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4), 32 GB each → ~30.57 GB usable/card |
| CPU | Ryzen 9 9900X (12c/24t); serving pinned to one CCD |
| Model | Qwen3.8-Flash-Next: 125.74B trunk + 51.2B PLE; 48 layers; 512 experts (10 routed + 1 shared); expert dim 640 |
| Token mixing | 3 × Gated DeltaNet + 1 × Qwen Sparse Attention per block of four |
| Hard weight budget | **GPU-resident weights ≤ 45.85 GiB** |
| PLE table | 51.2B params, streamed off-VRAM via Direct I/O (`--load-mode dio`, 0 MB VRAM) |
| Context / serving | 262,144 tokens, `--parallel 4`, `--kv-unified`, q4_0 KV |
| Quantizer | 1-pass `llama-quantize` + AtomicChat public imatrix (2.05M tokens, 0 starved experts) |

Two traps shaped everything:

1. **The `ncols = 640` trap.** `ffn_down_exps` is `[640, 2560, 512]`; 640 is not divisible
   by 256, so every 256-super-block quant silently falls back to `IQ4_NL` (+3.08 GiB VRAM
   for the 14 `Q2_0` layers). Only `Q2_0` (block 64) and `IQ4_NL` (block 32) are valid
   sub-8-bit quantizers for down-experts.
2. **The production load-fit trap.** At `--parallel 4` the runtime allocates per-slot compute
   graphs that single-slot (`-np 1`) runs never exercise — measured ~1.5 GiB on GPU 0 /
   ~3.1 GiB on GPU 1, plus KV and a ~1.2 GiB draft graph. A +225 MiB weight increase is
   enough to turn a comfortable load into a production OOM. The 45.85 GiB budget is fitted
   to that envelope, never extrapolated.

## 🧭 The development story — what was tested, what was learned

Every build was a *test*; each produced one *lesson*; each lesson drove the next
*increment*. The chart below is the complete record — **all twelve builds in three eras**,
size and headline metric on every node, one lesson per build — and the footer lists the
eleven levers that were tried, measured, and closed.

![The CodeUD lineage — all twelve builds in three eras with sizes, metrics and lessons, and the eleven falsified levers beneath](lineage.png)

### Era 1 — Finding the constraints (v3 → v9)

The project started at **142 GiB** with a broad per-tensor precision ladder (PLE `Q8_0`,
output head `Q6_K`, attention `Q5_K`/`Q6_K`). v3 proved the hardware geometry works —
31.6 GB/GPU bounded, zero spillover, no driver resets — and then a **3,000-turn autonomous
agentic stress test** found the failure modes at >120k context: long-range attention key
drift, 3D-spatial coordinate errors, and frontend DOM/state errors. The calibration corpus
was skewed to Python and prose; the failures lived everywhere else.

**v4** answered with a 6-pillar calibration corpus (adding 600 geometry-proof/VSR samples
and 800 web/DOM samples) and lifted attention to `Q6_K`. The targeted domains improved
(geometry 1.789, web_ui 2.0347); wiki slightly regressed — **calibration composition is a
real tradeoff**. This era also established the imatrix proxy method: a BF16 imatrix needs
>120 GB VRAM (impossible locally), so calibration runs on a ~58 GiB high-precision proxy
with `F32` router gating, keeping expert routing essentially identical to full precision.

**v5/v6** found 46 GiB of slack in the ladder — v4-level quality at two-thirds the file
size. **v7** proved shard-splitting is orthogonal to quality.

**v8 was the pivotal failure.** Its first compact rules format moved *all* down-expert
projections to `IQ4_NL` with an `IQ2_XXS` middle — and geometry PPL exploded from 1.7846
to **2.0321 (+27%)**. That single number is the most important empirical fact of the
project: **down-projection precision is the binding constraint on spatial and numeric
domains.**

**v9** was the strategic pivot: 100% GPU residency with **stratified** experts (boundary
layers `IQ3_S`, transitional `IQ2_S`, middle `IQ2_XXS`), explicit `IQ4_NL` down-experts
(dodging the silent fallback trap), `Q8_0` shared experts / QSA attention / output head —
and the PLE table at `Q8_0` on disk, streamed via Direct I/O at **0 MB VRAM**. Geometry
recovered to 1.797. Lesson: **the PLE format is independent of VRAM — a disk-only
tradeoff.**

### Era 2 — The reasoning/prose trade-off and the master (v10 → v14)

**v10** transferred ISTA-DASLab's externally-solved RCO layer allocation. The reasoning
gains were real — **AIME 2025 went 76.7% → 86.7%** — but it funded them by cutting
embedding and attention precision, and paid everywhere else (HumanEval 1.1421, prose and
long-context regressed). **v11** tried to synthesize v10's reasoning core with v9's
restored precision: it produced the **all-time best geometry score (1.5287**, from a `Q5_K`
output head) but regressed prose badly — and it changed allocation *and* PLE format in
the same build, making the result unattributable. That confound is why every build since
isolates **one variable**.

**v12** synthesized the lineage — but silently downgraded the PLE table to `Q4_0` to save
~24 GiB of disk, a change that was **never A/B tested in isolation** and then carried
forward through v14–v17.

**v14 is the master**: a strict zero-sum Pareto rebalance of v12 — 34 `IQ4_NL` + 14
`Q2_0` down-experts, 10 middle gate/up layers at `IQ2_S` funding the down lift, `Q8_0`
shared experts, `F32` router gates, `Q5_K` output head, `Q6_K` attention. Five straight
domain wins over v12, 45.852 GiB resident. **It is still the fallback artifact today.**

### Era 3 — One variable, isolated (v18a)

**v18a** is the test v12 never ran: v14's rules with **exactly one line changed** —
`per_layer_token_embd.weight=q8_0` (tensor 1223 of 1224, byte-verified on rebuild). The PLE
table is off-VRAM, so the change costs **0 MB of VRAM and +23.84 GiB of disk**.

On the original corpora it improved **all five domains** (p≈0.03 under a null). After the
test-set incident (see *Measured quality*) and the 2026-10-21 reset, it was rebuilt
byte-exactly and re-measured same-session against v14: code/wiki/long better, Web-UI and
geometry a wash within noise, aggregate geo-mean **2.0381 vs 2.0402**, no domain ≥2σ
worse. It is also **faster**: sharper trunk logits raise ngram draft agreement
(80.4 t/s at 94.5% acceptance vs v14's 74.9 t/s at 92.6%). Promoted to the active
serving symlink on 2026-10-21.

### The full build-by-build record

| Build | Size | Measured result | Lesson learned |
|---|---|---|---|
| **v3** | 142.07 GiB | code 1.0914, wiki 3.5318 | precision *and* calibration content both matter at depth (3,000-turn stress test: key drift, spatial + DOM errors) |
| **v4** | 142.07 GiB | geometry 1.7890, web_ui 2.0347 (6-pillar calibration) | domain-targeted calibration works; composition is a tradeoff; imatrix-via-proxy method established |
| **v5** | 96.32 GiB | v4-level quality at 2/3 size (geometry 1.7846) | the precision ladder had 46 GiB of slack |
| **v6** | 96.27 GiB | identical to v5 within noise | packaging/refinement — no quality lever |
| **v7** | 91.88 GiB + 25 shards | code 1.1159, geometry 1.5858 | shard-splitting is orthogonal to quality |
| **v8** | 96.95 GiB | **geometry 2.0321 ✗** (+27% vs v5; all-`IQ4_NL` down experts) | down-projection precision is the **binding constraint** on spatial/numeric domains |
| **v9** | 97.17 GiB | 100% GPU-resident; geometry 1.7973 | stratify the experts, never uniform them; PLE format is VRAM-independent |
| **v10** | 70.74 GiB | AIME 2025: 76.7% → **86.7%**; but HumanEval 1.1421 ✗ | external reasoning-core gains were real, but VRAM-cut funding traded prose away |
| **v11** | ~96 GiB | geometry **1.5287** (all-time best, `Q5_K` output head); wiki 4.0059 ✗ | confounded build (allocation + PLE changed together) — isolate one variable per build |
| **v12** | 73.31 GiB | code 1.1173, wiki 3.8441, geometry 1.7068 | PLE silently downgraded to `Q4_0`, never A/B tested in isolation |
| **v14** | 72.67 GiB | code 1.1122, wiki 3.7074, web_ui 2.0116\*, geometry 1.7569\*, long 3.6643 | strict zero-sum discipline + five-domain gating = best all-round (**fallback today**) |
| **v18a** | 96.53 GiB | geo-mean **2.0381** vs v14 2.0402; 1223/1224 tensors identical to v14 | the isolated PLE `q4_0`→`q8_0` probe answers v12's open question — **the winner** |

\* v14's web_ui/geometry figures are from the original evaluation corpora. Those two sets
were later rebuilt and every model re-baselined in one session (v14: 1.7335 / 1.3496);
numbers from the two eras are never mixed. See *Measured quality*.

### The graveyard — eleven levers, tested and closed

Between v15 and the close of the program, every *obvious* quality lever was built or
probed and measured. All of them failed. This is the most useful thing the project
produced.

| # | Lever | Measured result | Why it failed |
|---|---|---|---|
| 1 | v15 — merge 3 community imatrices (3.17M tokens) | all deltas inside error margins (≤0.0016) | the 2-bit grid is too coarse: extra tokens cannot flip integer bin assignments; calibration saturates at ~2M tokens |
| 2 | v16 — transplant ISTA's externally-solved allocation | worse on **all five domains** (wiki 4.0064, kill gate) | an allocation is **solver-conditional** — co-adapted to its quantizer and imatrix; external maps never transfer |
| 3 | v17 — diagonal-objective knapsack (own data, −15% error mass) | code improved, but wiki 3.7974 → kill gate | the diagonal proxy **mis-ranks** (down_exps: −7.6% diagonal vs +7.7% true Hessian); proxy cuts don't convert to PPL |
| 4 | §10-B — joint-S activation scaling (static probe) | runtime gain −1.7% avg, **3× below** the pre-registered 5% gate | predicted per-tensor gains (−10.7%/−12.2%) were in-sample, per-tensor-optimal inflations; no build wasted |
| 5 | PLE "imatrix" | impossible by construction | the PLE table is consumed by a `GET_ROWS` gather, never a matmul — there is no Hessian to collect |
| 6 | v18b — PLE `q8_0` + one down-expert lift (L45) | **best code/wiki/long ever measured** — but hard OOM at production config | `--parallel 4` allocates ~1.5/3.1 GiB per-card compute buffers that `-np 1` validation never exercises; quality is void without load-fit |
| 7 | v19/v19b — zero-sum down-expert swaps (8 and 4) | Web UI **+0.21 / +0.17 (≈7σ**, reproduced exactly) | the cuts landed on the **first MoE blocks** (L1/L2/L5/L11), the most bit-sensitive; static error share is not marginal sensitivity |
| 8 | H1 — gate/up-only zero-sum reallocation (guarded) | geometry **+3.08σ**, geo-mean 2.0406 vs 2.0381 | even with early-layer cuts forbidden, the late-layer moves that helped three domains killed the fourth |
| 9 | H3 — swap in a code-heavy imatrix | geo-mean 2.0505 (+0.61%) | the AtomicChat gold imatrix is already the right one; corpus rebalancing trades domains at 2-bit |
| 10 | H4 — pooled-H GPTQ-style error feedback (96 tensors, values only) | local error **−10–12%**, end-to-end PPL *worse* (code +0.40σ, wiki +0.92σ) | local Hessian-weighted MSE ≠ downstream perplexity; the correction moved error into router/expert-sensitive directions |
| 11 | HB — SmoothQuant up-branch fold (2 measured outliers, Δ0 bytes) | all five domains **Δ ≤ +0.0003** (≈0.01σ) | the imatrix already supplies the per-channel MSE weighting — the outlier/block-scale mechanism does not exist in 1-pass weight-only quantization |

### The lessons that generalize

1. **A proxy that improves is not evidence.** v17's diagonal objective, H4's 10–12% local
   error cut, and v19's error-mass arithmetic all improved their proxy and lost perplexity.
   The only metrics that ever decided anything were end-to-end five-domain PPL and the
   production load-fit gate.
2. **Allocations are solver-conditional.** The best external allocation available made
   everything worse here. Derive from your own measured sensitivity or don't bother.
3. **Calibration saturates.** 2.05M tokens identified the channel variances; 3.17M merged
   tokens changed nothing; a rebalanced corpus traded domains. Once the imatrix knows the
   variance, the 2-bit grid decides.
4. **Early MoE blocks are bit-sensitive.** Cuts on L1/L2/L5/L11 cost ≈7σ of Web UI. Every
   lift must be funded somewhere, and the cheap-looking early cuts are the expensive ones.
5. **Quality is void without load-fit.** The best-measured build of the entire program
   (v18b) was unusable because it never survived the `--parallel 4` compute-buffer
   envelope. Measure at the config you actually serve, during generation.
6. **Isolate one variable.** v11 changed two things and taught nothing. v18a changed one
   line and settled a question five builds had carried open.

The graveyard collapses to one conclusion: **the v14/v18a allocation is already a local
optimum with steep walls.** The single real win of the whole program was the one variable
nobody had isolated — the off-VRAM PLE table — which is v18a.

## ⚖️ How the bits are spent (the byte/error budget)

Per-tensor allocation driven by where the reconstruction error actually is:

| Family | GiB | % resident | Static error mass | Avg bpw |
|---|---:|---:|---:|---:|
| gate + up experts | 23.75 | 51.8 % | **94.6 %** | 2.53 |
| down experts | 18.02 | 39.3 % | 5.2 % | 3.84 |
| attention | 1.26 | 2.7 % | ≈ 0.1 % | mixed |
| hyper-connections | 1.20 | 2.6 % | ≈ 0 | BF16 / F32 |
| shared experts | 0.23 | 0.5 % | ≈ 0 | Q8_0 |
| SSM / output head / embeddings / other | 1.35 | 2.9 % | ≈ 0 | mixed |
| **Total** | **45.85** | **100 %** | **100 %** | **3.13** |

- **Shared experts** `Q8_0` — active on every token, cheap to keep precise.
- **Router gates** `F32` — routing fidelity preserved exactly.
- **DeltaNet SSM recurrent state** `F32`/`BF16`/`Q6_K` — long-range state tracking is not
  where bits are cut.
- **MoE gate/up experts** carry 94.6% of the error and are already at the 2-bit floor.

The v19 failure is the cautionary footnote: down experts hold 39.3% of bytes for 5.2% of
*static* error, but cutting them damages Web UI ≈7σ — a tensor's static error share is
**not** its marginal sensitivity. The allocation respects that.

## 🥊 Base model & community comparison

AtomicChat published the cleanest public reference for this model: several bit-rates scored
against **BF16 logits** on a held-out set (87 chunks @ 4096 ctx; BF16 reference PPL
**4.0445**):

| Build | bpw | PPL ratio vs BF16 |
|---|---:|---:|
| AD-5.00bpw | 5.00 | 1.026 |
| AD-4.27bpw | 4.27 | 1.026 |
| AD-3.84bpw | 3.84 | 1.102 |

At ~3.8 bpw the model degrades ~10% in PPL, flattening to ~2.6% past ~4.3 bpw. This build
sits at **3.13 bpw**, below that tested floor, so the per-tensor allocation spends its
scarce bits exactly where that measurement says they matter.

| Build | Trunk bpw | GPU-resident | Source |
|---|---:|---:|---|
| BF16 base (reference) | ~16 | >120 GB | Qwen |
| Unsloth `UD-Q4_K_XL` | 5.25 | 76.9 GiB | community |
| Myric `APEX` (MIDDLE) | 4.14 | 85.4 GiB | community |
| Unsloth `UD-IQ4_XS` | 4.13 | 60.4 GiB | community |
| AtomicChat `AD-4.27bpw` | 4.27 | 56.1 GiB | community |
| AtomicChat `AD-3.84bpw` | 3.84 | 45.8 GiB | community |
| **CodeUD `v18a` (this repo)** | **3.13** | **45.85 GiB** | this repo |

The higher-bpw community builds trade the guarantee this one makes: that weights, KV,
drafter and projector all fit the 64 GB envelope under `--parallel 4` at 262k context.
Measured, not assumed: `AD-4.27bpw` OOMs that envelope by ~8 GB, and `UD-IQ4_XS` cannot
even load at 100% GPU offload — they require expert-tiering (a second serving stack) or
more VRAM.

## 📊 Measured quality

Raw-token perplexity on fixed test sets (same binary, same files; lower is better).
Absolute PPL is not comparable across harnesses — these are our own numbers.

| Domain | v18a | v14 (fallback) |
|---|---:|---:|
| Algorithmic code (HumanEval prompts) | **1.1058** | 1.1122 |
| Linguistic prose (WikiText-2) | **3.7004** | 3.7074 |
| Web UI (TS/React/CSS, rebuilt set) | 1.7413 | **1.7335** |
| Geometry (VSR test split, rebuilt set) | 1.3512 | **1.3496** |
| Long-context (WikiText-2 @ 4096) | **3.6530** | 3.6643 |
| **Geometric mean** | **2.0381** | 2.0402 |

Same session, same binary, same test files, paired per-chunk error margins. On the
original (pre-rebuild) corpora, v18a improved **all five** domains with consistent sign
(p≈0.03); the code/wiki/long files are the originals with verified provenance (v18a's
per-chunk values reproduce exactly), while the Web UI and geometry sets were rebuilt
deterministically after a disk-cleanup incident and every model re-baselined in one
session. On the rebuilt sets those two domains are a wash within noise.

**Canonical Wikitext-2 raw** — the community-comparable harness documented in
[`benchmarks/METHODOLOGY.md`](blob/main/benchmarks/METHODOLOGY.md): the full standard
test file, ctx 2048, fixed flags, same binary, all runs same-session. Comparison policy:
**same envelope only** — this project's purpose is the best quality that *fits this
hardware at full context*, so the table admits community quants that hold the same
GPU-resident class. Anything that cannot serve the 64 GB envelope is out of scope
(measured, below).

| Model | Resident | PPL | ± | paired t vs v18a | s/pass |
|---|---:|---:|---:|---:|---:|
| AtomicChat `AD-3.84bpw` (IQ4_XS-M64) | 45.8 GiB | **4.5212** | 0.0264 | **−3.3** | 7.30 |
| **CodeUD v18a (this repo)** | 45.85 GiB | 4.5618 | 0.0267 | — | **2.65** |
| CodeUD v14 (fallback) | 45.85 GiB | 4.5691 | 0.0267 | +1.9 | 3.33 |

Paired per-chunk t-tests across all 145 chunks
([`benchmarks/paired_ttest.py`](blob/main/benchmarks/paired_ttest.py)); negative t means
the row model beats v18a per-chunk:

- `AD-3.84bpw` is genuinely ahead on prose — 0.9% per chunk — but pays **2.75× per-pass
  time** for it on identical silicon (the IQ4_XS-heavy mix hits slow dequant paths on
  gfx1201). Its 45.8 GiB figure is community-reported; the operational same-class proof
  is that it loads at the family-default `53,47` split like v18a/v14. It has never been
  load-tested at 262k / `--parallel 4`.
- v18a's edge over v14 on this single domain is inside noise (t = +1.9) — the five-domain
  record is the sharper instrument between the two.

**It is also faster, not just better:** sharper trunk logits raise ngram draft agreement —
80.4 t/s decode at 94.5% draft acceptance vs v14's 74.9 t/s at 92.6% (same prompt,
drafter, and tensor split).

**Out of envelope — measured, not assumed.** AtomicChat `AD-4.27bpw` (56.1 GiB resident)
needs a rebalanced `50,50` split even for a 2048-ctx perplexity load and cannot serve
this project's config: +10.25 GiB resident against the 2.07 GB v18a leaves free at the
production shape. Unsloth `UD-IQ4_XS` (60.4 GiB resident) fails even a bare
`llama-perplexity` load at 100% GPU offload — `cudaMalloc failed: out of memory`, 30.8 GiB
demanded on one card at `50,50`. Both are bigger quants with better perplexity and no way
to run the mission; their raw runs are archived in
[`benchmarks/results/`](tree/main/benchmarks/results) as the fit-failure record, not as
comparison rows. The BF16 reference cannot be run on this hardware either (>120 GB VRAM).

### Two harnesses — and why the second one exists

Everything both harnesses ever measured is shipped in
[`benchmarks/`](tree/main/benchmarks): all five-domain test sets with their provenance
record, the canonical WikiText-2 file, both runners, the raw result JSONs and the
per-chunk logs. The five-domain numbers above are only citable because the exact files
they were measured on are in the repo.

**Harness 1 — the five-domain internal gate** (2026-09 → 2026-10-21). Four raw test
sets, five measurements: algorithmic code (HumanEval prompts), prose (WikiText-2),
Web UI (CodeFeedback rows), geometry (VSR flattened to text) and long-context (the
WikiText-2 file again at ctx 4096). Same-session A/B only, paired per-chunk
verification, per-domain error margins, pre-registered kill gates. It exists because
a single prose metric is blind to a domain-specific collapse, and it earned its keep:
every lever in the graveyard above ran through it.

**Why it was replaced** — in order of severity:

1. **Subset evaluation hides in the basis.** The prose domain read the same canonical
   WikiText-2 test file, with the same binary and flags, as the standard harness — but
   stopped at chunk 32 of 145. The file's tail is harder than its head, so the running
   estimate climbs steadily from ~1.7 to ~4.6: at chunk 32 it reads **3.7004**, at chunk
   145 **4.5618** — same model, same bytes, same flags, **+0.86 PPL of pure evaluation
   depth**. Fine for A/B gating, where both sides of every comparison share the basis;
   unpublishable as an absolute number.

    ![Cumulative PPL over the canonical WikiText-2 test file — chunk 32 (internal gate basis) vs full file](benchmarks/ppl_chunks_wikitext2.png)

2. **Bespoke is not comparable.** Absolute PPL is only valid across models when file,
   flags and binary are identical. Nobody outside this project could run our five sets;
   the llama.cpp convention can be run by anyone.
3. **Two domains were the wrong instrument.** Geometry is a *visual* benchmark flattened
   into a rigid text template, and "code PPL" over HumanEval prompts has no executable
   meaning. Both served as sensitivity canaries — a multi-domain surface that reacts
   when prose doesn't — which is exactly how they were used.
4. **The gold standard was never available.** KL-divergence against BF16 logits — what a
   serious quantizer publishes — needs >120 GB of VRAM. On 64 GB that is not a benchmark,
   it is a hardware order. Every number here is quant-vs-quant, never quant-vs-reference.

The gate is not retired: five-domain A/B remains the internal accept/reject instrument
for any future candidate, and the standard full-file harness is what every publishable
number is measured with. The full methodology review that motivated the switch lives in
[`benchmarks/METHODOLOGY.md`](blob/main/benchmarks/METHODOLOGY.md); the harness map and
reproduction instructions live in
[`benchmarks/README.md`](blob/main/benchmarks/README.md).

## 🔭 What would change the answer

1. **A 2-bit format that llama.cpp can execute.** If a trellis/additive or learned-codebook
   format lands in GGML with RDNA4 kernels, the 45.85 GiB wall moves without changing the
   allocation. The closest concrete target is a 3-bit block-32 down-expert format filling
   the `Q2_0` ↔ `IQ4_NL` gap at `ncols = 640`. This is the only genuinely open lever left.
2. **Expert pruning to buy bits** — model surgery, not a quant.
3. **A production-config load-fit pass for any future candidate** — the v18b lesson is
   standing: `-np 1` validation does not predict `--parallel 4` fit.

At 45.85 GiB the model is already spending its bytes where its error is. The remaining
quality is not in *which* tensor gets *which* 2-bit type; it is in making the 2-bit type
better, or removing parameters to buy a higher one. **v18a is the best servable llama.cpp
artifact the current format set can produce.**

## 📚 References — the work this project stands on

### AtomicChat — the calibration is literally theirs

- The imatrix this GGUF ships with **is** AtomicChat's published
  [`imatrix.gguf`](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/imatrix.gguf),
  byte-identical (2.05M tokens, 926 entries), with the calibration corpus public at
  [`AtomicChat/calib-corpora`](https://huggingface.co/datasets/AtomicChat/calib-corpora).
  Two dedicated experiments (v15 merge, H3 swap) failed to beat it.
- Their documentation of **N-gram PLE table isolation** and the **`ncols: 640` quantizer
  fallback** shaped this project's PLE handling and the v9 residency pivot.
- Their published **BF16-referenced benchmarks** (KL-divergence against raw BF16 weights,
  computed on 4× B200) are the source of the PPL-ratio table in the comparison section.
- Their `AD-3.84bpw` / `AD-4.27bpw` builds are the community reference points in the
  comparison and the same-harness runs.

Repo: [`AtomicChat/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)

### ISTA-DASLab — allocation and Hessian research

- Their **GSQ-RCO** per-tensor allocation research directly motivated two experiments
  here: **v10** (layer-allocation transfer → AIME 76.7% → 86.7%) and **v16** (full
  manifest transplant → worse on all five domains). The general lesson — *an allocation
  is solver-conditional* — came out of their work meeting our 1-pass quantizer.
- Their **Hessian-aware error-compensation research line** (GPTQ) motivated **H4**, the
  pooled-H value-feedback experiment (built, verified, closed end-to-end).
- In the v12-era comparison, their GSQ-RCO `IQ3_XXS` build out-PPL'd ours on every
  measured domain (wiki 3.3510 vs 3.8441) — solver quality this project's 1-pass
  `llama-quantize` pipeline could not match at equal bytes. That gap is exactly what the
  new-format path above exists to close.

Repo: [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)

### Research literature

Papers that map onto specific experiments in this card (references verified 2026-10-20):

| Work | Where it shows up in this project |
|---|---|
| **Qwen3.8-Next** — [arXiv:2608.30320](https://arxiv.org/abs/2608.30320) | the base architecture: GDN/QSA hybrid, PLE at layer 2, which tensors are foldable |
| **GPTQ** — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | H4 — pooled-H GPTQ-style error feedback (built, closed) |
| **AWQ** — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | §10-B — joint-S activation scaling (probed, closed) |
| **SmoothQuant** — [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) | HB — up-branch outlier fold (built, closed) |
| **QuantMoE-Bench** — [arXiv:2406.08155](https://arxiv.org/abs/2406.08155) | predicts the v19 failure: first MoE blocks deserve more bits |
| **GEMQ** — [arXiv:2605.23078](https://arxiv.org/abs/2605.23078) | expert-level mixed-precision principles (the v17 objective) |
| **QuIP#** — [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · **AQLM** — [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) · **QTIP** — [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) | the 2-bit format frontier — the only genuinely open lever |

### The stack

- **Qwen team** — the base model and its license.
- **llama.cpp / ggml** — the GGUF format, quantization types and kernels this whole
  project is built on.
- **[stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)** —
  the RDNA4 fork this serves on (BF16 HC-mixer fusion, GDN chunking, QSA fixes).
- **Unsloth** and **Myric** — community quants used as comparison points.

## 🚀 Run it

Targets the RDNA4 ROCm fork (`stew675/llama-cpp-rdna-boosts`, r87) on dual R9700s.

```bash
llama-server \
  -m Qwen3.8-Flash-Next-CodeUD-v18a.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf \
  --model-draft mtp-Qwen3.8-Flash-Next-shared-IQ4NL-Q3K.gguf \
  --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 \
  --n-gpu-layers 49 --split-mode layer --tensor-split 53,47 --main-gpu 0 \
  --ctx-size 262144 -ctk q4_0 -ctv q4_0 -b 1024 \
  --flash-attn on --load-mode dio
```

<details>
<summary>Required environment (ROCm 7.2.4, gfx1201)</summary>

```bash
export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export HSA_FORCE_P2P_COPY=1
export HSA_ENABLE_SDMA=1
export GPU_POWER_PROFILE=COMPUTE
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=16
export ROCBLAS_USE_CLIENT_WORKSPACE=1
export ROCBLAS_USE_HIPBLASLT=1
export HIPBLASLT_WORKSPACE_SIZE=134217728
export GGML_CUDA_ALLREDUCE=hybrid
export GGML_CUDA_GDN_CHUNKED=1
export GGML_CUDA_GDN_CHUNKED_BF16=1
export GGML_CUDA_GDN_ALIGN_BOUNDARY=1
export LLAMA_LAZY_IO_THREADS=4
export LLAMA_QSA_SPARSE_FA=1
export LLAMA_QSA_DENSE_SHORTCUT=1
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
```

Pin the process to one CCD of the 9900X (`taskset -c 0-5,12-17`); cross-CCD scheduling
costs measurable decode throughput.

</details>

What to expect, measured on the reference hardware:

- Single-slot (`-np 1`) with the MTP drafter: **~80 t/s** decode at ~94% draft acceptance.
- Multi-slot Studio serving (`--parallel 4`, 262k, `--kv-unified`): 40–55 t/s. VRAM is
  **fitted to the edge by design** — expect ~1.4 GB free on GPU 0 and ~0.7 GB on GPU 1 at
  peak, sampled during generation. That envelope is the budget, not an accident.
- `--load-mode dio` is mandatory: it streams the PLE table from NVMe and keeps host RAM
  (and the display GPU that shares it) out of the picture.

## 📦 Files

| File | Size | Purpose | Status |
|---|---:|---|---|
| `Qwen3.8-Flash-Next-CodeUD-v18a.gguf` | 96.53 GiB | quantized weights (45.85 GiB resident + 50.7 GiB off-VRAM PLE `q8_0`) | ✅ uploaded — SHA-256 below |
| `mmproj-Qwen3.8-Flash-Next-F16.gguf` | 0.84 GiB | vision projector (F16) | ✅ in this repo |
| `mtp-Qwen3.8-Flash-Next-shared-IQ4NL-Q3K.gguf` | 1.18 GiB | speculative-decoding drafter | ✅ in this repo |
| `lineage.png` | 583 KB | the full twelve-build lineage chart above | ✅ in this repo |
| `benchmarks/` | ~3 MB | both harnesses (`METHODOLOGY.md`, `README.md`), all five-domain test sets + provenance + rebuild script, the canonical WikiText-2 file, both runners + paired t-tests, raw result JSONs + per-chunk logs (including the community reference runs), the evaluation-depth chart | ✅ in this repo |

SHA-256 of `Qwen3.8-Flash-Next-CodeUD-v18a.gguf`:
`9d62569c698e8a544824e77082649dbf13f73a4d3aabc766dffad7448552b1ab`

## 📄 License

Qwen Community License 1.0, inherited from the base model — the full text is the
[`LICENSE`](blob/main/LICENSE) file in this repo.