---
library_name: llama.cpp
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - rocmfpx
  - gfx1151
  - strix-halo
  - qwen35
  - dense
  - mtp
  - gated-deltanet
  - vulkan
  - Q3_0_ROCMFPX
pipeline_tag: text-generation
language:
  - en
  - multilingual
version: '1.0'
date: 2026-08-23
quantized_by: pugant
---

# Qwen3.8-27B — Q3_0_ROCMFPX base + agent (Strix Halo / gfx1151)

## TL;DR

`Qwen3.8-27B` — the dense hybrid-attention Qwen release (48 gated-delta-net layers + 16
full-attention layers, `full_attention_interval = 4`, native 262K context, `qwen35` GGUF arch) —
quantized in **two arms** with the ROCmFPX fork FP3 presets:

- **base** — `Q3_0_ROCMFPX`, 4.44 bpw effective, 14.125 GiB
- **agent** — `Q3_0_ROCMFPX_AGENT`, 5.72 bpw effective, 18.198 GiB

**MTP layer included** (`blk.64` with `nextn.*` tensors, `nextn_predict_layers = 1`): serve with
`--spec-type draft-mtp` to enable speculative decoding. Built for **AMD Strix Halo (gfx1151)** on
the **ROCmFPX fork family** — we serve and benchmark these files on **our lab runtime** (full
source: [pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa/tree/main/rocmfpx);
upstream: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)).

> ⚠️ Both GGUFs use **GGML type 104** (`q3_0_rocmfpx`) and are for the **ROCmFPX fork of
> llama.cpp**. They will not load in stock llama.cpp (`invalid ggml type`).

## Honest verdict — read this first

On this hybrid architecture **these presets are not a true 3-bit**. The fork routing protects
**ALL attention tensors — including the 48 fused `attn_qkv` of the gated-deltanet layers — with
K-quants**, so the effective density is **4.44 bpw (base, 14.13 GiB)** and **5.72 bpw (agent,
18.20 GiB)**. Neither arm beats the ROCmFP4 baseline where it matters: **base** is dominated by
STRIX_LEAN on all three axes (worse ppl, −15.6% tg128, +2.2% size — see
[pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN));
**agent** does buy back ppl (+1.89%/+2.04% vs BF16, better than LEAN's +2.74%/+3.42%) but pays
with 5.72 bpw (larger than the FULL preset) and −31.7% tg128 — worse quality-per-byte and
quality-per-second than the FP4 presets.
This repo is published as **reproducible research data** (preset comparison under matched conditions)
and for the **type-104 GGUFs themselves**.

## ⚠️ Critical warnings — read before use

- **Fork-specific tensor types** (`q3_0_rocmfpx`, `q6_0_rocmfpx`, K-quant attention protection).
  **Requires a ROCmFPX fork build with GGML type 104 support** — see [Runtime](#runtime). Upstream
  [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) loads these files too (any recent
  build with the custom GGML types — HIP or the Vulkan-only build).
- **AMD RDNA 3.5 (gfx1151 / Strix Halo) target.** Tested on Radeon 8060S iGPU, not elsewhere.
- FP3 here is a **memory-bandwidth play** that did not materialize: as the verdict above says, the
  effective density lands at 4.44–5.72 bpw, not below the FP4 baseline.

## Benchmarks

Hardware: AMD Ryzen AI Max+ 395, Radeon 8060S (gfx1151, 128 GB unified LPDDR5X), exclusive GPU,
Vulkan RADV, 2026-08-23.

Perplexity method: `llama-perplexity -c 512 -b 512 -fa on -ngl 999`, wikitext-2-en 150k tokens +
Italian technical corpus 51.5k tokens. BF16 baseline 6.6409 (en) / 11.7156 (it). The STRIX_LEAN
control was reproduced within sigma on the same backend (Vulkan offset +0.34% / +0.49%).

| Arm | Preset | eff. bpw | Size | PPL en (Δ vs BF16) | PPL it (Δ) | tg128 | pp512 |
|---|---|---:|---:|---:|---:|---:|---:|
| STRIX_LEAN (baseline) | Q4_0_ROCMFP4_STRIX_LEAN | 4.38 | 13.82 GiB | 6.8226 (+2.74%) | 12.1168 (+3.42%) | 13.06 ± 1.34 | 340.96 ± 6.09 |
| base (this) | Q3_0_ROCMFPX | 4.44 | 14.125 GiB | 6.9943 (+5.32%) | 12.3121 (+5.09%) | 11.02 ± 1.47 (−15.6%) | 264.89 ± 5.16 (−22.3%) |
| agent (this) | Q3_0_ROCMFPX_AGENT | 5.72 | 18.198 GiB | 6.7665 (+1.89%) | 11.9541 (+2.04%) | 8.92 ± 1.01 (−31.7%) | 298.74 ± 10.63 (−12.4%) |

tg/pp = `llama-bench -p 512 -n 128 -fa 1 -ngl 999 -r 5`, same session; tg/pp deltas vs LEAN
in-session; quality deltas vs BF16 historical references, with the LEAN control within sigma.

Notes from the table:

- **base** is strictly dominated: larger than the baseline (14.125 vs 13.82 GiB), worse perplexity
  on both corpora, and slower on both axes (−15.6% tg128, −22.3% pp512).
- **agent** buys back quality (+1.89% / +2.04% vs BF16, the closest of the three to BF16) but pays
  with 5.72 bpw effective, 18.198 GiB and −31.7% tg128.
- Structural cause below: the 3-bit block type never reaches the attention mass on this hybrid
  architecture, so the density advantage is spent before it starts. The FP4 STRIX_LEAN preset
  remains the Pareto pick on gfx1151.

## Routing histogram (post-quant verification)

866 tensors per arm, **0 routing errors, 0 NaN/Inf** (post-quant verification over every tensor):

| Type | base | agent |
|---|---:|---:|
| Q3_0_ROCMFPX | 313 | 235 |
| Q6_K | — | 123 |
| Q5_K | 115 | 42 |
| Q4_K | 42 | — |
| Q6_0_ROCMFPX | 33 | 105 |
| Q4_0_ROCMFP4_FAST | 2 | — |
| Q8_0 | 1 | 1 |
| F32 | 360 | 360 |
| **Total** | **866** | **866** |

In the base arm the `Q8_0` tensor is `nextn.eh_proj` and the two `Q4_0_ROCMFP4_FAST` tensors are
the MTP draft FFNs — the same tensor mapping as the STRIX_LEAN build. The 360 F32 tensors are the
small tensors left untouched. Both histograms sum to 866 with no unclassified tensor.

## Usage (fork build)

```bash
llama-server -m Qwen3.8-27B-Q3_0_ROCMFPX-base.gguf \
  -ngl 999 -fa on --jinja -c 32768 \
  --host 0.0.0.0 --port 1234
# + MTP speculative decoding:
#   --spec-type draft-mtp --spec-draft-n-max 4 \
#   --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
```

Swap `-m` for `Qwen3.8-27B-Q3_0_ROCMFPX-agent.gguf` to serve the agent arm.

> Requires a **ROCmFPX fork build with GGML type 104 support** (recent Vulkan or HIP builds) —
> see [Runtime](#runtime).

## Importance matrix

Same self-produced importance matrix as the STRIX_LEAN build, calibrated on an interleaved corpus
(agentic-coding traces from
[ProCreations/grug-think-v3-10k](https://huggingface.co/datasets/ProCreations/grug-think-v3-10k),
Italian prose, real code). The matrix is published for reuse:
[pugant/Qwen3.8-27B-imatrix](https://huggingface.co/pugant/Qwen3.8-27B-imatrix)
(works with any llama.cpp `llama-quantize --imatrix`).

## Quantization details

- **Presets:** `Q3_0_ROCMFPX` (base arm) and `Q3_0_ROCMFPX_AGENT` (agent arm), quantized in the
  fork convert container.
- **The "3.50 bpw nominal" caveat.** The preset is nominally 3.50 bpw; the effective density on
  this architecture is **4.44 bpw (base)** and **5.72 bpw (agent)**. Structural explanation: in the
  fork's quantize routing the attention branch (`category_is_attn_v`) covers the **fused
  `attn_qkv` tensors of all 48 gated-deltanet layers** and routes them to K-quants instead of the
  FP3 block type (Q5_K/Q4_K in base; Q6_K/Q5_K in agent — visible in the histogram above), and the
  `use_more_bits` imatrix boost lifts further tensors. With only the FFN mass left for the 3-bit
  type, the nominal density never materializes.
- **Source:** BF16 GGUF from `unsloth/Qwen3.8-27B-GGUF`, SHA-verified at quant time by the
  STRIX_LEAN pipeline (same source shards).
- **imatrix:** the self-produced matrix of the STRIX_LEAN pipeline, loaded with **496/496 entries**
  (the LEAN coverage-gate count: all 64 non-MTP layers covered, 0 NaN/Inf).
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). MTP
  layer included (`blk.64`, `nextn_predict_layers = 1`).

## Acknowledgements

- **[Alibaba / Qwen team](https://huggingface.co/Qwen)** for the base model
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) and for training and
  open-sourcing the MTP layer that makes speculative decoding possible on this class of dense
  models.
- **[Unsloth](https://huggingface.co/unsloth)** for the BF16 GGUF release used as quantization
  source (SHA-verified by the pipeline).
- **[charlie12345](https://github.com/charlie12345)** and all the contributors of
  [ROCmFPX](https://github.com/charlie12345/ROCmFPX) — the llama.cpp fork that defines the
  ROCmFPX presets (the Q3_0_ROCMFPX family quantized here and the ROCmFP4/STRIX_LEAN baseline),
  MTP speculative serving, and the Vulkan kernels for gated-delta-net / SSM ops that this hybrid
  architecture requires.
- **[kyuz0](https://kyuz0.github.io/amd-strix-halo-toolboxes/) (Donato Capitella)** for the amd-strix-halo-toolboxes —
  the Dockerfiles (ROCm and Vulkan fork builds) used for quantization and serving in this work.
- The **Strix Halo / gfx1151 community** (strix-halo-guide, RADV bench threads) whose
  cross-backend benchmarking methodology informed our measurement protocol.

## License

Apache-2.0 (inherited from the base model). See LICENSE.

---

## Files in this repo

| File | Size |
|---|---:|
| `Qwen3.8-27B-Q3_0_ROCMFPX-base.gguf` | 14.125 GiB (15,166,725,120 bytes) |
| `Qwen3.8-27B-Q3_0_ROCMFPX-agent.gguf` | 18.198 GiB (19,540,229,120 bytes) |

## Runtime
**Benchmark environment:** one bare-metal AMD Strix Halo (Ryzen AI MAX+ 395, 128 GB) — full dated configuration and measurement policy: [BARE-METAL.md](https://github.com/pugant/strix-nebulosa/blob/main/BARE-METAL.md)

Requires a ROCmFPX fork build (custom tensor types — stock llama.cpp refuses the file).
**Recommended: our lab build** ([pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa), `main`) —
reasoning budget and persistent prompt cache on every model; drafter features
where the model ships one: see the engine section of its README.

Kept as a documented NO-GO (not a true 3-bit on this arch); if you want small, use the LEAN above.

> Everything here is experimental and provided as-is, at your own risk.
