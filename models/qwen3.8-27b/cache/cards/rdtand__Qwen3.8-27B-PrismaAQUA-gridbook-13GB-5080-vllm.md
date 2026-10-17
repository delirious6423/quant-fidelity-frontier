---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - prismaquant
  - aqua
  - gridbook
  - codebook-quantization
  - mixed-precision
  - vllm
pipeline_tag: text-generation
library_name: vllm
---

# Qwen3.8-27B — AQUA, 13 GB, built for a 16 GB card

**A 13 GB download: 12,982,426,694 bytes = 12.98 GB = 12.09 GiB.** Weights and
codebooks are 12,958,885,485 B (12.069 GiB); the tokenizer and configs add the
remaining ~23 MB. On a 16 GB (= 16 GiB) consumer Blackwell card (RTX 5080-class)
that leaves **~3.9 GiB** for CUDA context, KV cache and activations.

> **On the two size numbers.** 13 GB and 12.1 GiB are the same bytes in
> different units — decimal GB (what your disk and the Hub report) and binary
> GiB (what `nvidia-smi` and the allocator report). This model is named for the
> **13 GB** figure because that is the number you have to fit, and because a
> "12.1" in the title reads as headroom that is not there. Every table below
> states bytes exactly so neither unit has to be trusted.

Every Linear in this model was given its *own* format and its own codebook size,
chosen by **AQUA** — an allocator that prices each candidate against the loss it
actually costs, rather than assigning one uniform precision to the whole
network. The result is a 19-rung ladder used unevenly on purpose:

    355 FP8_CB_K28    94 FP8_CB_K48    20 NVFP4_CB_K16     8 NVFP4_CB_K12
      8 NVFP4_CB_K14   6 NVFP4_CB_K18    4 FP8_CB_K32       1 FP8_CB_K40

| | |
|---|---|
| weights + codebooks | **12,958,885,485 B** = 12.959 GB = **12.069 GiB** |
| bits per param, body Linears | **3.604** (10.971 GB over 24.351 B params) |
| bits per param, all allocated units | **3.855** (12.959 GB over 26.893 B params) |
| whole repo | **12,982,426,694 B** = 12.98 GB = 12.09 GiB — configs add ~23 MB |
| serving | stock vLLM + the **GridBook 0.8.8** out-of-tree plugin |
| embedding / head | `model.embed_tokens` → NVFP4 · `lm_head` → FP8 dynamic |
| removed | MTP heads, visual tower (text-only checkpoint) |
| context | 262144 |

**Two bit-rates are quoted because one number cannot mean both things**, and
publishing only the flattering one is how bpp labels go bad. The arithmetic is
here so it can be checked against the checkpoint:

| | bytes | params | bpp |
|---|---|---|---|
| 496 body Linears | 10.971 GB | 24.351 B | 3.604 |
| `model.embed_tokens` | 0.715 GB | 1.271 B | 4.500 |
| `lm_head` | 1.272 GB | 1.271 B | 8.006 |
| **all 498 allocated units** | **12.959 GB** | **26.893 B** | **3.855** |

The 3.604 figure follows this project's convention of reporting over
quantizable parameters and excluding the head; 3.855 is what the whole
checkpoint costs. The allocator's own predicted figure for its body solve was
3.3236 — that is a *recipe* number over a slightly different denominator, it is
not reproducible from these bytes, and it is not quoted as this artifact's
bit-rate. Note also that bpp labels are not comparable across this project's
own accounting eras.

> **This is not a vanilla-vLLM artifact.** PrismaQuant's `compressed-tensors`
> lane serves on unmodified vLLM with no plugin; a *codebook* format does not —
> it needs GridBook's kernels. If you want a no-plugin artifact, use the
> `compressed-tensors` releases instead.

## Installing GridBook

GridBook is on PyPI. It is an **out-of-tree vLLM plugin**, so where you install
it matters more than the command:

```bash
pip install gridbook==0.8.8
```

**That is the exact wheel this artifact was validated on.** PyPI's
`gridbook-0.8.8-py3-none-any.whl` has sha256

```
a982e8842d0ce183eaad8978941a375dc984fa0697be7c4519dd741efd1153a3
```

which is byte-identical to the wheel that served every gate recorded in
`shipcard.json` — the eager and graph load+generate gates, the ship gate, and
the KL/PPL measurements below. So this is not "a compatible version": installing
from PyPI gets you the bytes the numbers were measured on. Verify it yourself:

```bash
pip download --no-deps gridbook==0.8.8 -d /tmp/gb && sha256sum /tmp/gb/*.whl
```

**Install it into the same environment as vLLM.** vLLM discovers it through the
`vllm.general_plugins` entry point (`gridbook = "gridbook:register"`); a plugin
in a different venv is simply never found, and the model then fails to load
with an unknown quantization method rather than with a useful error.

**Requirements**

| | |
|---|---|
| Python | 3.10 – 3.13 |
| OS | Linux |
| deps | `torch`, `safetensors`, `huggingface-hub` — all **unpinned on purpose** |
| GPU | NVIDIA; kernels are compiled for your device's exact compute capability |
| build tool | **`nvcc` (a CUDA *toolkit*, not just a runtime)** |

The dependencies are deliberately unpinned because vLLM's torch is usually a
local-version wheel (e.g. `2.13.0+cu130`) that no PyPI pin can satisfy. If your
resolver nonetheless tries to replace torch, install without deps — vLLM
already provides all three:

```bash
pip install --no-deps gridbook==0.8.8
```

**`nvcc` is required, and this is the step people miss.** The wheel is
`py3-none-any`: the CUDA decode and prefill kernels are **compiled on first
use** through `torch.utils.cpp_extension.load` and pinned to your GPU's exact
architecture with a single `-gencode`. A CUDA *runtime* install has no `nvcc`,
and the failure surfaces at first forward, not at install time. Check first:

```bash
nvcc --version          # must print a version
```

If it does not, either install a CUDA toolkit and point at it —

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # a directory containing bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
```

— or get the compiler from PyPI:

```bash
pip install nvidia-cuda-nvcc-cu13
```

**Give the build cache a persistent, writable directory.** Otherwise every
serve recompiles the kernels:

```bash
export PRISMAQUANT_CB_EXT_DIR=~/.cache/gridbook-ext    # default: ~/.cache/prismaquant-*
```

Expect the **first** serve to take noticeably longer than later ones; that is
the one-time compile, and it is cached per (GPU architecture, GridBook build).

**Verify before you serve:**

```bash
python -c "import gridbook; print(gridbook.__version__)"
python -c "from importlib.metadata import entry_points; \
  print(list(entry_points(group='vllm.general_plugins')))"
nvcc --version
```

The second command must list a `gridbook` entry. If it does not, the plugin is
installed in the wrong environment.

## Serving

```bash
vllm serve rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

GridBook registers the codebook quantization method through vLLM's out-of-tree
quantization plugin interface; vLLM itself is unforked and unpatched, and
nothing here patches it at runtime.

**The exact stack the ship gates ran on**, stated because "it works on my
machine" is not a serving claim:

| | |
|---|---|
| vLLM | `0.26.1rc1.dev693+g7f7a32cfe.d20260812` |
| torch | `2.13.0+cu130` |
| GridBook | `0.8.8` (PyPI) |
| GPU | NVIDIA GB10 (DGX Spark), Blackwell `sm_121` |

That vLLM is a **development build**, not a PyPI release — do not try to `pip
install` that exact version string, it will not resolve. GridBook declares no
vLLM version bound, so any vLLM new enough to expose the quantization-plugin
interface should load this artifact; **no other vLLM version has been tested
against it**, and a load failure on a different build is a plugin-interface
mismatch rather than a problem with these bytes.

This artifact was built and validated on a 128 GB unified-memory GB10, not on a
discrete 16 GiB card. The 16 GiB claim above is an arithmetic one — 12.069 GiB
of weights inside a 16 GiB frame — and the headroom figure is what is left over,
not a measured maximum context length on that card.

## What AQUA actually decided

**454 of 496 body units (91.5%) land in the A8 family even though the fp4
codebook rungs are cheaper in bytes** — only 42 units (8.5%) take an A4 rung.
That is AQUA pricing the A4 *activation* contract, and it is the whole reason
this lane needed AQUA rather than a weight-only cost table: NVFP4 and NVFP4A16
render **bit-identical weights** and differ ~9.4% RMS on activations, so a
weight-side objective is provably indifferent between them and cannot see the
A4↔A8 boundary at all.

### The layer map

![Per-Linear format allocation across all 64 layers](allocation-map.png)

One column per layer, one row per projection; every cell is one Linear,
colored by the format AQUA chose for it, rendered from this artifact's own
`quant_config.json`. Teal = FP8 codebook rungs, brighter with codebook size
K; navy = NVFP4 codebook rungs. The hybrid architecture is visible directly:
full attention (`self_attn.*`) exists only on every fourth layer, linear
attention everywhere else. The A4 units concentrate where they are cheapest
to give up — early-layer MLP projections — while the `in_proj_z`/`in_proj_a`
gates hold the brightest (K48) rungs.

The same map is browsable cell-by-cell, alongside every other PrismaQuant
artifact, in the [allocation explorer](https://prismaquant.org/explorer.html).

## Honest accounting

This is the first AQUA-on-codebook artifact ever built, so the quality of the
cost fit is part of what is being published, not a footnote.

### Two units the allocator could not price

`model.embed_tokens` and `lm_head` have no probe row and no cost row on this
lane, so they cannot enter the allocator's DP. They were assigned afterwards,
the recipe records that explicitly (`__prismaquant__.aux_assignments_added`),
and neither was guessed — both were chosen on **exact full-vocab measured KL**
on the real 248320×5120 tensors, a strictly stronger basis than the surrogate
the DP uses for the body:

| unit | option | size | measured KL | |
|---|---|---|---|---|
| `model.embed_tokens` | **NVFP4 group-16** | **0.715 GB** | **0.001063** | **chosen** |
| | FP8 per-row | 1.272 GB | 0.000342 | |
| | INT8 W8A16 | 1.272 GB | 0.000331 | |
| `lm_head` | **FP8 per-row** | **1.272 GB** | **0.001647** | **chosen** |
| | NVFP4 W4A4 | 0.715 GB | 0.015553 | |
| | split fp8+nvfp4 | 0.771 GB | 0.003000 | not built — needs a row-split head |

The embedding would pay 0.557 GB for 0.00073 KL to move to FP8, far worse than
the body's rate at this budget. The head is the mirror image: NVFP4 costs
0.0156 KL to save the same 0.557 GB. Two caveats, stated rather than smoothed:

* **`lm_head` could not have been a codebook rung here even in principle.** A CB
  rung on the head exports cleanly and then dies at load (`no module or
  parameter named 'lm_head.cb_qweight'`) because no GridBook method claims
  `ParallelLMHead`. Shape legality is not servability. The head's only servable
  choices are the delegated stock ones, which is what the table compares.
* The **row-split head** is the one known, quantified, unbuilt improvement:
  0.501 GB smaller for +0.00135 KL.

The allocator was handed a deliberately inflated budget so that removing those
two units from the DP was an exact change of variables
(16.099 − 2.543 − 2.543 = 11.013 GB of body; 13.000 − 0.715 − 1.272 = 11.013 GB),
and the re-pricing came out with 1.30 MB of slack against the true 13.0 GB
budget — which is what confirms the substitution was exact rather than
approximately right. Note that bpp labels are not comparable to this project's
own earlier ones: the accounting convention has changed across eras.

### The cost fit did not validate — and what that is worth

The anchored cost path prices a 19-rung ladder from a small number of rendered
anchors plus a sampled panel. Its own held-out gate returned
**`BAD_FACTORISATION_SIGNAL`**: 111 of 192 validation cells over the 0.05 dex
bar, max |dex| 0.594, and `extrapolated_fraction_of_all_units` = 0.9879.

Two of those cells are design errors in the validation itself, named rather than
buried: `FP8_CB_K36` was **both** the fp8 anchor and an fp8 validation rung, so
its dex is 0.0000 by construction and 48 of the 192 cells validated nothing; and
NVFP4-CB was validated only at K12 and K24, the two extremes, which measures the
worst extrapolation rather than the typical one. On the three rungs that
genuinely generalize: fp8 K44 median 0.1233 / max 0.4398; nvfp4 K12 median
0.1103 / max 0.5944; nvfp4 K24 median 0.1016 / max 0.4665. (*dex* is a base-10
log ratio — a median of 0.11 is a factor of ~1.29, not 11%.)

"The fit has error X" is not the decision-relevant question: the DP **ranks**
cells, and error that is common-mode within a segment cancels in the ranking. So
the error was priced directly — resample the *empirically observed* dex errors
from this campaign's own held-out rows (anchor-rung zeros excluded, pool of 144,
median 0.1125, max 0.5944), apply 10^(±dex) per cell, re-solve the DP, and
re-score every resulting assignment on the one true cost table:

| arm | unit churn | family churn (A4↔A8) | true Δloss regret |
|---|---|---|---|
| full observed fit error, 4 seeds | 16.5–20.0% | **2.2–4.4%** | **+6.3 … +8.8%** |
| weight-only (no AQUA) | 54.4% | **39.5%** | **+74.4%** |

Baseline 3.3236 bpp, true predicted Δloss 0.07354. Errors are drawn
independently per cell, which is deliberately pessimistic — real fit error is
correlated within a segment, and correlated error cancels. The entire observed
fit error is worth **≤8.8% of the objective and ≤4.4% of the A4/A8 decision**;
AQUA's own signal is ~10× larger on both axes.

**The caveat, because it is the honest half:** +74.4% is scored under AQUA's
*own* objective. It establishes that the two arms **disagree strongly** — not
that AQUA is **right**. Only a served KL A/B settles that, and this artifact
does not contain one. What the churn measurement settles is the narrower fork it
was run to settle: *the fit's imprecision cannot explain the gap, so the gap is
signal.*

### Serving-metric results

<!-- GOLD:BEGIN -->
Measured on the **served artifact** — this exact checkpoint, loaded by vLLM
through GridBook, at the stack pinned above. Both arms of the KL were served in
the same session at the same top-K, with `--kv-cache-dtype auto` on **both** so
the reference is a true BF16 reference and not an fp8-KV approximation of one.
The teacher is the unquantized BF16 Qwen3.8-27B text checkpoint.

| | measured | how |
|---|---|---|
| **KL vs BF16, confident positions** | **0.05617** | `kl_confident_mean` over the 2108 of 4088 positions where the BF16 teacher's top-1 mass exceeds 0.5 |
| KL vs BF16, all positions | 0.09167 | `kl_mean`; floor-inflated wherever teacher mass runs past the top-K window, so it is reported, not quoted |
| worst single position | 2.762 | `kl_max` — the tail is real and is not hidden behind the mean |
| top-1 agreement, confident | **96.87 %** | the artifact picks BF16's argmax at 2042 of those 2108 positions |
| top-1 agreement, all positions | 86.47 % | |
| top-K window / coverage | K = 1024 · mean 98.71 % · min 50.92 % | `topk_coverage_mean` is the teacher probability mass actually captured |
| **WikiText-2 perplexity** | **9.792** vs **9.365** BF16 → **+4.56 %** | direct, on the served artifact; 8176 tokens scored at seqlen 512 |
| mean NLL | 2.2815 vs 2.2370 BF16 → +0.0446 nats | |
| worst chunk mean NLL | 2.7042 vs 2.6783 BF16 → +0.0259 nats | the PPL tail degrades *less* than the mean |
| sampling contract | n = 8 × seqlen 512 (KL) · 16 × 512 (PPL) | this project's canonical gold contract |
| spec-decode | not detected | a spec-decode-on serve returns the *draft* model's logprobs and would silently poison both numbers |

Provenance is in the record, not just in this table: `gold_kl.json` carries the
build commit (`1ccdf58`), the calibration-contract hash, and the serve
fingerprint, and both gold slots are bound into `shipcard.json`.

**What kind of KL this is.** A served `/v1/completions` returns top-K prompt
logprobs, never the full 248,320-way distribution. This is therefore a **top-K
KL with a declared tail**, not the exact full-vocab KL that PrismaQuant's
`compressed-tensors` lane quotes — a codebook artifact only exists behind a
vLLM plugin, so there is no offline full-vocab path to it. Teacher and student
are served at the identical K and the record carries it, which is what makes
the two comparable to each other; it does not make either comparable to a
full-vocab number from another lane.

**Do not compare 0.056 to PrismaQuant's other published KLs.** Two things
differ at once, and both matter more than the digits: those artifacts are at
**5.31 and 4.75 bpp** while this one is at **3.855**, and they were measured by
a different evaluator on a different lane. A lower bit-rate buying a higher KL
is the *expected* shape of the trade — this artifact exists to fit a 16 GiB
card, and 12.069 GiB is not free. The comparison that would be meaningful is
against another quantization **at this size**, which is not something this card
can assert because the measurement has not been run.

**The BF16 perplexity reference was measured, not looked up.** 9.365 comes from
serving the unquantized BF16 text checkpoint through the *same* tool, corpus,
seqlen and token budget — 8176 tokens scored on both arms, `--kv-cache-dtype
auto` on both — so the +4.56 % is a like-for-like delta rather than this
artifact's absolute PPL set against a number from somewhere else. An absolute
WikiText PPL is nearly meaningless as a quantization claim; the delta is the
claim. **A 27B model at 12.069 GiB — roughly 4.4× smaller than its ~54 GB BF16
source — gives up 4.56 % perplexity.**

**What the numbers say, plainly.** On positions where BF16 is confident, this
artifact agrees with its argmax 96.9% of the time; on the ~3% where it does
not, and in the `kl_max` = 2.76 tail, it genuinely diverges. Mean KL is a
summary and this project has repeatedly found that a good mean can hide a heavy
tail, so both are printed above. For tool-calling and other
single-decision-point workloads, the tail — not the mean — is the number to
weigh.
<!-- GOLD:END -->

## Provenance

Every number above is either recomputed from the artifact's own bytes or carried
in machine-checked records inside the repo:

* `shipcard.json` — the refusal contract: artifact digest, build commit, and one
  record per ship gate (native-export eager/graph load+generate on the pinned
  serving stack, the catastrophic-quality gate, and the gold KL/PPL records).
  Publication refuses on an unverified card.
* `quant_config.json` — per-Linear format assignment, the codebook references,
  and the render provenance.
* `cb_codebooks.pqcb` — the codebooks themselves, digest-bound to the config
  groups that reference them.

## Citation / contact

PrismaQuant — Robert Tand, independent researcher — <robert.tand@icloud.com>
