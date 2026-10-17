---
language:
- en
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- qwen3.8
---

# Qwen3.8-Flash-Next-VQ-2.1bpw

**44.5 GiB — a 335 GiB frontier MoE on 64 GB machines.**

A data-free vector-quantized build of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(180B total / 10-of-512 active, 51.2B n-gram PLE, vision) for Apple
Silicon. Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint as `model.py`. Built with [VQLab](https://github.com/noahzelezny/VQLab).

MoE experts at d=8/K=16384 (14-bit codes, padded-tail packed) for gate/up, d=4/K=256 for every down_proj (exact packing), PLE n-gram tables at d=8/K=256 (8-bit rows); layers 0–1 at d=2/K=256 and the KL-chosen band 27–33, 35, 47 at d=4/K=256 gate/up (v2, see the changelog).

The affine builds compared against below are our own conversions made with
the same tooling, scored on the same instrument.

![where these releases sit](chart_ladder.png)

## Requirements

**Requires Knurlogic 0.1.3 or newer.** Earlier Knurlogic releases, and plain
`mlx-lm` builds of `qwen4_exp` that rebuild the PLE hash multipliers from the
default seed, hash tokens into the n-gram embedding tables with the wrong
multipliers and read the wrong rows on every token. The model still loads and
generates, but its output is degraded.

**Under [Knurlogic](https://github.com/noahzelezny/Knurlogic) none of this section applies:** Knurlogic registers the `qwen4_exp` architecture itself (see Run it). The rest of this section is for plain `mlx-lm`.

**This model needs an `mlx-lm` that has the `qwen4_exp` architecture, which no
released version has yet.** The architecture is in
[ml-explore/mlx-lm PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788),
still unmerged as of 2026-09-09. On a stock `pip install mlx-lm` you
will get:

```
ModuleNotFoundError: No module named 'mlx_lm.models.qwen4_exp'
```

That is the architecture missing, not a problem with this artifact. Until the
PR merges:

```bash
pip install git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1788/head
```

The VQ runtime itself needs no patches — it ships inside the checkpoint as
`model.py` and stock `mlx-lm` executes it. The PR is required only for the
base architecture, and only without Knurlogic.

**Two or more Macs:** Knurlogic splits one model across two or more Macs. Install the same
knurlogic version and the model on each, run `knurlogic ui --host cluster` on each, and
launch the model from the page with the machines selected. Knurlogic drafts with
`mtp-head-q6.safetensors` when the model has it (see Speculative decoding).
Budget ≈2.2 GiB extra resident when drafting is enabled. On a 64 GB
machine the model alone is 45.3 GiB with its vision tower, so the head fits but leaves little
headroom for anything else. Measured on this rung through exo, the earlier
serving path (M4, single node): 23.7–25.5 tok/s decode at acceptance 0.82–0.88.

## Changelog

### 2026-10-04 — rescored on Knurlogic 0.1.3

Knurlogic 0.1.3 fixes how Qwen3.8-Flash-Next hashes tokens into its per-layer n-gram embedding (PLE) tables: earlier runtimes rebuilt the hash multipliers from a wrong default seed instead of using the values stored in the checkpoint, so every build of this model (and Qwen's own weights) read the wrong embedding rows on every token. This model now requires Knurlogic 0.1.3 or newer for correct output. The weights and the bundled `model.py` are unchanged; every number in Measured results was re-measured on the fixed runtime.

### 2026-10-01 — runtime refresh; Knurlogic

The bundled `model.py` is updated to the current VQ runtime, and the v2 weights described in the 2026-09-17 entry are published with this revision for the first time. The run, multi-Mac, drafting and vision instructions now use [Knurlogic](https://github.com/noahzelezny/Knurlogic).

### 2026-09-17 — v2 codebooks: per-layer allocation chosen by KL

**Weights changed.** 46 down_proj demoted d8-K16384 → d4-K256 (exact packing, −1.7 GB of padding), gate/up promoted d8-K16384 → d4-K256 on L27, 28, 29, 30, 31, 32, 33, 35, 47; 45.3 GiB (v1: 45.8). Size 45.3 GiB. The codebooks are the
same data-free k-means fits; only WHICH layers carry the wider tier changed,
and it was chosen by `vqlab alloc-sweep` on the three house corpora at 12288 tokens (F100), re-scored on the paired three-corpus KL instrument 2026-09-17.

KL to the bf16 teacher, millinats per token, paired on identical positions,
three house corpora at 12288 tokens (|t| > 2 is a real difference):

| corpus | v1 | v2 | change | t |
|---|---|---|---|---|
| prose | 409.6 | 383.9 | -6.3% | -7.3 |
| code | 109.4 | 100.1 | -8.5% | -6.1 |
| literary | 356.9 | 334.0 | -6.4% | -9.2 |

Measured on the runtime before Knurlogic 0.1.3; not comparable with Measured results below.

Perplexity on the same runs, printed with its sign and NOT used as a gate (it disagrees with KL in sign on prose for this family — F116–F118): prose 5.8255 → 5.7485, code 1.7243 → 1.7128, literary 7.8240 → 7.6574.

Runtime: this bundle carries the v2 runtime profile (bf16 I/O on; F53/F105),
+11% prefill / +19% decode over the arc6 runtime published 2026-09-09.

### 2026-09-09 — runtime refresh

**Runtime refresh.** The bundled `model.py` is updated so downloaders run
exactly the code that was benchmarked; dense bundles now carry both runtimes.

- Faster prefill on affected geometries, measured per rung: a
  device-codebook kernel arm for large-codebook geometries (up to 1.46x on
  affected rungs), a ragged-subvector relaxation (up to 1.34x on affected
  rungs), a fused d8 arm, and a routing memo (≈2%). No blanket speedup is
  claimed across the lineup — gains apply only where the geometry engages
  the new paths.
- Output quality is unchanged: the kernel changes are bit-identical or
  1-ULP-equivalent, and the routing memo is bit-identical (logits checksum
  verified).
- Speculative decoding (MTP): on repos that ship
  `mtp-head-q6.safetensors`, the sidecar works with the exo fork branch
  `mtp-stage1` (github.com/noahzelezny/exo) — launch each node with
  `exo --mtp` (or set `EXO_MTP=1`). Note: with MTP enabled, exo serves
  requests sequentially (the batch engine has no MTP path), so leave it
  off for concurrent / multi-agent workloads.

### 2026-09 — bundle refresh

**Bundle refresh.**

`model.py` is updated:

- Faster MoE expert kernels (threadgroup-occupancy fix, device-direct
  activation reads, template-free dispatch, and a simd_sum reduction).
  Measured on this family: decode 17.4 -> 18.8 tok/s (+8%, A-B-A, M3
  Ultra). All kernels verified bit-identical to the previous bundle
  except the reduction, which is 1-ULP equivalent with a measured
  quality delta of exactly zero (identical perplexity to 16 digits,
  KL 0.0 between old and new logits). `VQ_D8_SS=0` restores the
  bit-identical legacy reduction.
- The runtime caps MLX's buffer-reuse cache by default
  (`VQLAB_CACHE_LIMIT_GB` overrides, `0` disables), so long-prompt
  prefill no longer balloons transient memory — peak stays near
  resident instead of a multiple of it.
- The bundle loads under both `mlx-lm` and `mlx_vlm`.
- **Nothing you have downloaded breaks**: weights unchanged, only
  `model.py` and `config.json` keys replaced; prior revisions stay
  pinnable by commit hash.

**Peak memory, measured externally:** process RSS sampled at 5 Hz from
outside across load + 2048-token prefill + 128-token decode peaks at
**45.8 GiB**; the runtime returns transient prefill
allocations as they free, so peak ≈ resident (without the MTP sidecar;
add ≈2.2 GiB when enabled).

## Measured results

Every row re-measured 2026-10-04 on Knurlogic 0.1.3 (mlx 0.32.3). KL in
millinats per token against the bf16 teacher's cached top-64 logits, 12288
tokens per corpus on three corpora (prose, code, literary); every row is
scored on the same cache and the same positions. Sizes are text weights; the VQ builds
also carry a 0.84 GiB vision tower, and every build ships a 2.1 GiB MTP head. Numbers published before 2026-10-04 were measured on
runtimes that hashed the PLE tables wrongly (see the changelog) and are not
comparable to these. KL is higher than in those earlier tables because the
fixed teacher is much sharper (prose perplexity 2.31); the ordering of builds
is unchanged.

| build | size | prose KL | code KL | literary KL | mean KL | top-1 (prose) | ppl (prose) |
|---|---|---|---|---|---|---|---|
| affine q3 (ours) | 75 GiB | 2027.4 | 730.0 | 1180.0 | 1312.4 | 49.5% | 14.054 |
| **this model** | **44.5 GiB** | **896.8** | **276.8** | **783.7** | **652.4** | **72.3%** | **3.559** |
| VQ-3.2bpw | 68.7 GiB | 305.6 | 81.8 | 54.6 | 147.3 | 86.7% | 2.300 |
| affine q4 (ours) | 96 GiB | 624.1 | 169.3 | 101.1 | 298.2 | 77.8% | 3.912 |
| VQ-4.4bpw | 93.6 GiB | 100.0 | 29.5 | 13.5 | 47.7 | 92.2% | 2.323 |
| affine q5 (ours) | 116 GiB | 178.4 | 88.4 | 25.7 | 97.5 | 89.0% | 2.549 |
| VQ-5.5bpw | 111.6 GiB | 68.8 | 16.5 | 7.3 | 30.9 | 93.7% | 2.244 |
| affine q6 (ours) | 137 GiB | 84.1 | 26.9 | 8.7 | 39.9 | 93.4% | 2.276 |
| affine q8 (ours) | 178 GiB | 50.4 | 12.1 | 4.8 | 22.4 | 94.8% | 2.367 |
| bf16 teacher | 335 GiB | 0 | 0 | 0 | 0 | 100% | 2.31 |

**At this size there is no affine build to pair against**: the smallest,
affine q3, is 75 GiB. VQ-2.1bpw is the lowest-fidelity build in the table and
the only one that fits a 64 GB machine.

**Rank by KL, not perplexity.** KL and top-1 order this ladder as the bit
budgets predict; perplexity does not. VQ-3.2bpw reads 2.300 on prose, lower
than VQ-4.4bpw (2.323) and affine q8 (2.367), while sitting at six times q8's
KL. Perplexity is a mean over finite text and absorbs offsetting errors; KL
measures the distribution itself.

## Speed

Measured 2026-10-01 on this revision.

One Mac Studio M3 Ultra, `vqlab speed-pair`: a fresh process per run,
alternating with the reference, 3 pairs, 2048-token prompt, 128 generated
tokens, no drafting.

| | this model | Qwen3.8-Flash-Next-3bit |
|---|---|---|
| decode | **16.4 tok/s** (0.82×) | 19.6 tok/s |
| prefill | 849 tok/s (0.87×) | 981 tok/s |
| peak memory | 50.6 GB | 83.7 GB |

Throughput differs across Apple silicon; compare the ratio, not the absolute numbers.

## Run it

Requires a Mac with Apple Silicon (macOS, MLX).

With [Knurlogic](https://github.com/noahzelezny/Knurlogic), which works out
the settings and checks the model fits before loading it:

```bash
pip install "knurlogic>=0.1.3"
hf download TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw --local-dir ~/Knurlogic/Models/Qwen3.8-Flash-Next-VQ-2.1bpw
knurlogic serve ~/Knurlogic/Models/Qwen3.8-Flash-Next-VQ-2.1bpw
```

Chat at http://127.0.0.1:8080/, or point any OpenAI-, Anthropic- or
Ollama-compatible client at it (model `local`). `knurlogic ui` opens a page
with a Launch button for every model on disk instead.

Images work through Knurlogic directly; `mlx-vlm` is not needed.

The VQ runtime ships inside this repo as `model.py` (declared by
`model_file` in `config.json`), and Knurlogic runs the file the model ships.

Without Knurlogic:

```bash
pip install git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1788/head

python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 512
```

## Speculative decoding (MTP)

This repo includes `mtp-head-q6.safetensors` (2.1 GiB): the model's own
multi-token-prediction head, quantized. Stock loaders ignore it — it
costs nothing on disk-to-RAM unless you opt in by name.

**When enabled it adds ≈2.2 GiB resident** (head weights + its cache) on
top of the trunk, so budget for it. What you get, measured on this
artifact (M3 Ultra, greedy, 378-token runs): 18.1 -> 24.4 tok/s,
acceptance 0.77. The trunk verifies every drafted token by exact
rejection sampling, so the output distribution is exactly the base
model's. Prefill in the speculative path is chunked (2048-token chunks,
`VQLAB_PREFILL_CHUNK` overrides) with per-chunk eval, so long prompts
stay within the same memory envelope as plain decoding.

Knurlogic loads the head and drafts automatically (its `lean` preset turns
MTP off). To serve it with VQLab instead:

```bash
git clone https://github.com/noahzelezny/VQLab && cd VQLab
python3 -m venv .venv && source .venv/bin/activate
pip install .
python -m vqlab.cli serve \
  --model TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw \
  --sidecar mtp-head-q6.safetensors
```

OpenAI-compatible API on localhost; `vqlab mtp-generate` for one-shot
CLI use. Without `--sidecar`, nothing about the model changes.

## Methodology

Fitted **data-free** from the bf16 checkpoint — k-means / Lloyd over weight
subvectors, seed 1234, no Hessian, no activation statistics, no calibration
corpus. Recipes are in the VQLab repo.

Quantization damage is not uniform across layers. A one-pass probe
(teacher and student streamed together, per-layer local damage measured
with no compounding) shows the same hot set on every rung of this family:
layer 1 dominates, a late band (31–39) follows, and the map is identical
across geometries (rank correlation 0.905, identical top-10). Upgrading
only those layers buys 15–24% KL for 3–4% size on the lower rungs; the
probe, the mixing, and the verdicts are all reproducible with VQLab
(`vqlab layer-leverage`, scatter fits via `fit-moe --vq-layers`).

## Verification

Release gates passed on this artifact before upload: file, index and
tokenizer checks, a verbatim match between the bundled runtime and its
source, and a generation smoke through the shipping runtime on Apple
Silicon. The upload path runs the gate itself and refuses to publish
without it.

## Limitations

- **Requires Knurlogic 0.1.3 or newer.** Older runtimes, and plain `mlx-lm`
  builds that rebuild the PLE hash multipliers from the default seed, read the
  wrong n-gram embedding rows. See Requirements.
- **Plain `mlx-lm` needs an unreleased version.** The `qwen4_exp` architecture is
  in [PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788), still unmerged
  as of 2026-09-09. Knurlogic does not need it. See Requirements.
- **Rank the table by KL, not perplexity.** Perplexity is an aggregate over
  finite text and absorbs offsetting errors; several rungs read within noise of
  the teacher on perplexity while differing by an order of magnitude in KL.
- **The affine comparators are our own conversions**, not community builds.
- **Perplexity is not comparable across model families** — only within this
  table, which is one instrument on one corpus set.

## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/data-free-vector-quantization)

## Support this work

VQLab and these artifacts are built and released independently — the fits,
the measurement harness and the published rungs are one person's compute and
time. If they are useful to you:

- **Sponsor:** [github.com/sponsors/noahzelezny](https://github.com/sponsors/noahzelezny)
- **Contract work:** available for quantization and on-device inference work
  on Apple silicon — custom rungs, per-layer allocation for your model, or
  getting a checkpoint to run well on a Mac.
  Contact: [hello@thedrainflorist.com](mailto:hello@thedrainflorist.com)

## Provenance

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — released under the **Qwen Community License 1.0**.
This is a quantized derivative and inherits that license; using it
means accepting the base model's terms.
Quantization: TheDrainFlorist, 2026.

Local artifact: `qwen4exp_vq_packed_mixL01`.

Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab).
