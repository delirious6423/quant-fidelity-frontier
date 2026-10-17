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
  - Q4_0_ROCMFP4_STRIX_LEAN
pipeline_tag: text-generation
language:
  - en
  - multilingual
version: '1.0'
date: 2026-08-15
quantized_by: pugant
---

# Qwen3.8-27B — Q4_0_ROCMFP4_STRIX_LEAN (Strix Halo / gfx1151)

## TL;DR

`Qwen3.8-27B` — the dense hybrid-attention Qwen release (48 gated-delta-net layers + 16
full-attention layers, `full_attention_interval = 4`, native 262K context, `qwen35` GGUF arch) —
quantized to **`Q4_0_ROCMFP4_STRIX_LEAN`** (4.34 BPW effective, 13.82 GiB). **MTP layer included**
(`blk.64` with `nextn.*` tensors, `nextn_predict_layers = 1`): serve with
`--spec-type draft-mtp` to enable speculative decoding. Tuned for **AMD Strix Halo (gfx1151)** on
the **ROCmFPX fork family** — we serve and benchmark these files on **our lab runtime** (full
source: [pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa/tree/main/rocmfpx);
upstream: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)).

> ⚠️ This GGUF is for the **ROCmFPX fork of llama.cpp**. It will not load in stock llama.cpp
> (`invalid ggml type`).

## ⚠️ Critical warnings — read before use

- **Fork-specific tensor types** (`q4_0_rocmfp4`, K/V protection, Q5_K token embeddings).
  **Requires a ROCmFPX fork build** — see [Runtime](#runtime). Upstream
  [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) loads these files too (any build
  with the custom GGML types — HIP or the Vulkan-only build, both ship the deltanet kernels:
  `gated_delta_net.comp`, `ssm_scan.comp`, `ssm_conv.comp`).
- **AMD RDNA 3.5 (gfx1151 / Strix Halo) target.** Tested on Radeon 8060S iGPU, not elsewhere.
- FP4 is a **memory-bandwidth play** on this class of hardware (RDNA 3.5 has no FP4 silicon):
  smaller weights → fewer bytes per token on the 270 GB/s unified LPDDR5X.

## Benchmarks

Hardware: AMD Ryzen AI Max+ 395, Radeon 8060S (gfx1151, 128 GB unified LPDDR5X), exclusive GPU
window (no other services), 2026-08-15. Method: plain = `llama-bench -ngl 999 -fa on -p 512 -n 128`;
MTP = `llama-server --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75
--spec-draft-p-split 0.10` + server timing (`timings.predicted_per_second`), 2 runs after warm-up.

| Test | Qwen3.8-27B (this) | Qwen3.6-27B | Backend |
|---|---:|---:|---|
| plain tg128 | **13.77 ± 0.02** | 13.51 ± 0.40 | ROCm |
| plain pp512 | **354.7** | 344.1 | ROCm |
| MTP n-max 2 — prose | **18.4 / 19.1** | 17.2 / 17.2 | ROCm |
| MTP n-max 2 — deterministic | **27.3 / 22.0** | 20.2 / 20.5 | ROCm |
| plain tg128 | 8.97 | 8.92 | Vulkan RADV |
| plain pp512 | 319.8 | 324.8 | Vulkan RADV |
| MTP n-max 2 — prose | 14.1 / 13.1 | 14.2 / 13.6 | Vulkan RADV |
| MTP n-max 2 — deterministic | **19.9 / 17.0** | 16.5 / 17.3 | Vulkan RADV |

### MTP n-max sweep (this file, p_min 0.75, 2 runs per prompt, same window)

| n-max | ROCm prose | ROCm deterministic | Vulkan prose | Vulkan deterministic |
|---:|---:|---:|---:|---:|
| plain | 13.8 | 13.8 | 9.0 | 9.0 |
| 2 | 18.4 / 19.1 | 27.3 / 22.0 | 14.1 / 13.1 | 19.9 / 17.0 |
| **4 (balanced pick)** | 19.4 / 17.1 | 38.4 / 22.6 | 17.7 / 18.4 | 35.5 / 19.0 |
| **6 (structured pick)** | 18.1 / 17.6 | **41.9 / 30.9** | 18.4 / 17.2 | **43.7 / 25.6** |

Per-position acceptance: on prose, positions 3+ are rarely accepted (0.21/0.09 at n-4) → prose
plateaus at ~18–19 tok/s for any n-max. On deterministic/structured content acceptance stays high
deep into the draft: n-4 best task (0.98, 0.93, 0.91, 0.86), n-6 (0.99, 0.90, 0.86, 0.83, 0.82,
0.80) — mean accepted length 6.2 — which is where 43.7 tok/s (3.2× plain decode) comes from.
Recommendation: **n-max 4** as default, **n-max 6** for code/structured/RAG workloads. (Unlike the
Qwen3.6-35B-A3B base where n-4 was dominated — optimal n-max is model-specific.)

### Head-to-head vs Qwen3.6-27B (same preset, same window, same method)

Qwen3.8-27B wins the MTP match everywhere: **+25–35% deterministic / +8–11% prose on ROCm**,
+20% deterministic on Vulkan, prose tied. Plain decode is identical and bandwidth-bound.

Notes:
- Plain decode of both dense 27B models is **bandwidth-bound and identical** (~13.8 tok/s on ROCm):
  the deltanet/attention mix does not change the ceiling; MTP is the speed lever (as with
  Qwen3.6-27B).
- On dense 27B models **ROCm beats Vulkan on decode** (13.8 vs 9.0 tok/s, +53%) — the opposite of
  the fp4 MoE-35B class where the Vulkan fork wins tg (81.6 vs 71.2). Pick the backend by model
  class: dense → ROCm, MoE fp4 → Vulkan fork. pp512: ROCm +11%.
- MTP acceptance (n-max 2, p_min 0.75): 0.81–0.99, mean accepted length 2.3–2.9 tokens,
  per-position (pos-1 0.85–0.99, pos-2 0.39–0.95) — the pos-2 draft of Qwen3.8 is unusually
  strong on predictable content (0.93–0.95), which is where the deterministic-content win comes from.
- MTP speedup over plain (ROCm): ~2.0× deterministic, ~1.35× prose.

## Usage (fork build)

```bash
llama-server -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 -fa on --jinja -c 32768 \
  --host 0.0.0.0 --port 1234
# + MTP speculative decoding (n-max 4 balanced; 6 for structured/code):
#   --spec-type draft-mtp --spec-draft-ngl all --spec-draft-n-max 4 \
#   --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
```

## DFlash2 block-diffusion drafter (optional, our lab build)

Besides the built-in MTP layer, this model works with the external **DFlash2**
drafter ([incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF),
1.9B, Q4_K_M ≈ 1.1 GB) via our lab build (full source included under
[`rocmfpx/`](https://github.com/pugant/strix-nebulosa/tree/main/rocmfpx)) — port of upstream PR
[#27342](https://github.com/ggml-org/llama.cpp/pull/27342); usage notes in
`rocmfpx/docs/dflash2.md`.

```bash
llama-server -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 -fa on --jinja -c 16384 \
  --spec-type draft-dflash \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75
```

**Route by workload** (Vulkan RADV, temp 0, single stream, p_min 0.75):

| Workload | MTP n6 | DFlash2 n7 | pick |
|---|---|---|---|
| deterministic (counting/alphabet) | 45.2 / 26.1 | **57.4 / 36.3** | DFlash2 (+27–39%) |
| repetitive code | 27.5 | **33.8** | DFlash2 (+23%) |
| fixed-format logs | 28.0 | **35.8** | DFlash2 (+28%) |
| structured JSON | 35.0 | 36.2 | either (~floor) |
| free prose | **19.6 / 20.2** | 14.2 / 15.0 | MTP |

n-max 5 is the best single-drafter compromise (det 52.2/39.5, prose 17.5/15.9).
The DFlash2 acceptance stays ≥0.5 per position up to position 7 on structured
content, where MTP collapses after position 1; on free prose the acceptance is
on par with MTP but the block-diffusion round has a fixed cost, so MTP wins.

## Importance matrix

The quant was built with an importance matrix calibrated on an interleaved
corpus (agentic-coding traces from
[ProCreations/grug-think-v3-10k](https://huggingface.co/datasets/ProCreations/grug-think-v3-10k),
Italian prose, real code). The matrix is published for reuse:
[pugant/Qwen3.8-27B-imatrix](https://huggingface.co/pugant/Qwen3.8-27B-imatrix)
(works with any llama.cpp `llama-quantize --imatrix`).

## Quantization details

- **Type:** `Q4_0_ROCMFP4_STRIX_LEAN` — fork preset (4.38 BPW nominal): FP4 single-scale blocks
  for most matrices, **attention K/V protected**, **Q5_K token embeddings**. Effective 4.34 BPW on
  this architecture (13.82 GiB, 866 tensors, MTP included: `nextn.eh_proj` → q8_0, draft FFNs →
  `q4_0_rocmfp4_fast`).
- **Source:** BF16 GGUF from `unsloth/Qwen3.8-27B-GGUF` (2 shards, SHA256-verified against the HF
  LFS OIDs before quantization).
- **imatrix: self-produced.** Unsloth's calibration matrix for Qwen3.8 is **not published**
  (referenced in their quant metadata, absent from the repo). We generated our own with
  `llama-imatrix` (fork, GPU offload, 256 chunks × 512 ctx, ~10 min) over a 1.7 MB composite,
  round-robin-interleaved corpus: ~55% agentic-coding transcripts (tool-calls + code),
  ~28% Italian technical prose, ~17% real source code.
- **imatrix coverage gate (passed):** 496 tensor entries — exactly the same count unsloth's
  imatrix references for this model (metadata `quantize.imatrix.entries_count = 496`). All 64
  non-MTP layers covered (attention + gated-deltanet SSM tensors), 0 NaN/Inf, all entry shapes
  consistent. By design no imatrix can cover: MTP layer `blk.64` (not executed in a plain forward
  pass), `ssm_conv1d` (no observer hooks), `token_embd`/`output` — these take the unweighted path,
  identical to unsloth's behaviour on Qwen3.6.
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
  The unsloth GGUFs are text-only (no vision tensors); the vision tower is available separately
  (`mmproj-F16.gguf` in their repo, untested here).

## Acknowledgements

- **[Alibaba / Qwen team](https://huggingface.co/Qwen)** for the base model
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) and for training and
  open-sourcing the MTP layer that makes speculative decoding possible on this class of dense
  models.
- **[Unsloth](https://huggingface.co/unsloth)** for the BF16 GGUF release used as quantization
  source (SHA-verified) and for their calibration-matrix practice, which we replicated with a
  self-produced imatrix after finding theirs unpublished for this model.
- **[charlie12345](https://github.com/charlie12345)** and all the contributors of
  [ROCmFPX](https://github.com/charlie12345/ROCmFPX) — the llama.cpp fork that defines the
  ROCmFP4 family, the STRIX_LEAN preset, MTP speculative serving, and the Vulkan kernels for
  gated-delta-net / SSM ops that this hybrid architecture requires.
- **[kyuz0](https://kyuz0.github.io/amd-strix-halo-toolboxes/) (Donato Capitella)** for the amd-strix-halo-toolboxes —
  the Dockerfiles (ROCm and Vulkan fork builds) used for quantization and serving in this work.
- The **Strix Halo / gfx1151 community** (strix-halo-guide, RADV bench threads) whose
  cross-backend benchmarking methodology informed our measurement protocol.

## License

Apache-2.0 (inherited from the base model). See LICENSE.

---

## Runtime
**Benchmark environment:** one bare-metal AMD Strix Halo (Ryzen AI MAX+ 395, 128 GB) — full dated configuration and measurement policy: [BARE-METAL.md](https://github.com/pugant/strix-nebulosa/blob/main/BARE-METAL.md)

Requires a ROCmFPX fork build (custom tensor types — stock llama.cpp refuses the file).
**Recommended: our lab build** ([pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa), `main`) —
reasoning budget and persistent prompt cache on every model; drafter features
where the model ships one: see the engine section of its README.

The canonical model of the lab: dual-drafter routing was measured here (agentic +19%); MTP layer included.

> Everything here is experimental and provided as-is, at your own risk.
