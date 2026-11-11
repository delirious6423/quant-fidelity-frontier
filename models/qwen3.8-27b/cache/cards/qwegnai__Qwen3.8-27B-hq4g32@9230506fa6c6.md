---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: qwegn
pipeline_tag: text-generation
tags:
  - qwegn
  - apple-silicon
  - metal
  - quantized
  - hq4
  - dense
  - latent-sparse-attention
  - long-context
---

# Qwen3.8-27B · `HQ4_G32_M5` · Qwegn native pack

A native weight pack for the **[Qwegn inference engine](https://github.com/qwegnai/qwegn)**, a Swift + Metal 4 runtime for Apple Silicon. LSA holds prefill near **850–876 tok/s from 10K to 1M tokens**, so a 262K prompt finishes **2.45×** sooner than with dense attention.

<p align="center"><img src="media/qwegn_model_qwen38_27b.png" alt="Qwen3.8-27B + DSpine-b16 on Qwegn" width="960"></p>

<p align="center"><img src="media/qwegn_lsa_scaling.png" alt="Qwen3.8-27B on Qwegn with LSA, 64K to 1M tokens" width="960"></p>

> This pack uses Qwegn's own `QWP1` container and loads only in Qwegn. It is not a safetensors, GGUF or MLX checkpoint.

Pair it with [`qwegnai/Qwen3.8-27B-DSpine-b16-hq4g32`](https://huggingface.co/qwegnai/Qwen3.8-27B-DSpine-b16-hq4g32) (**134.5 tok/s peak / 100.5 tok/s geomean**, `4.19×` serial) or [`qwegnai/Qwen3.8-27B-DFlash2-hq4g32`](https://huggingface.co/qwegnai/Qwen3.8-27B-DFlash2-hq4g32) for block speculative decoding.

## At a glance

| Property | Value |
| :--- | :--- |
| Upstream model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Parameters | 27.78 B |
| Architecture | 64-layer dense hybrid: 48 Gated-DeltaNet + 16 full-attention layers, 5120 hidden, 24Q / 4KV GQA, MTP head, vision tower |
| Pack size | 18.40 GiB |
| Weights | `HQ4_G32_M5` + BF16, 5.69 bits/param |
| Native context | 262,144 tokens; inference extension tested near 1M with YaRN + LSA |
| LSA | ✅ `model.qlsa` sidecar included |

## Performance on Qwegn

Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core GPU, 585 GB/s), macOS 26+; `llmprobe v0.6.16–v0.6.17`, `--kv-mode q4 --temperature 0`.

| Metric | Result |
| :--- | :---: |
| Prefill (64K `e-means` / 10K prompt) | **914 tok/s** (`850 tok/s` @ 10K) |
| Cold TTFT | 94–113 ms |
| Warm prompt-cache TTFT | 6–12 ms (307×) |
| Standard decode (serial) | 31.2–32.9 tok/s |
| Speculative decode (DSpine-b16 drafter) | **100.5 tok/s geomean** (**134.5 tok/s** peak, `4.19×`) |
| Speculative decode (Variational MTP-3 / DFlash 2) | 84.4–91.3 tok/s / 83.8 tok/s |
| 1M-token prefill / decode (`e-means-4096` / `Ward-512`) | **831 / 35.6 tok/s** (`805 / 36.5` `Ward-512`) |
| Needle recall 64K → 1M | **3 / 3 at every length** |
| Core reasoning eval (`10Q`) / Code eval (`100Q`) | **8 / 10 (80%)** · **73 / 100** |

Short-context results are from separate workloads; warm-cache TTFT and the listed cold TTFT are not a controlled same-prompt ratio. Speculative short-context runs use Variational MTP-3 (margin 2.5, probability floor 0.2, hybrid drafting), or DFlash 2 for the drafter row. Variational acceptance can change output versus exact greedy decoding. Code evaluation, where shown, uses 50 HumanEval + 50 MBPP tasks with reasoning disabled. Results vary with workload and thermals.

## Long context with Latent Sparse Attention (`Ward-512` vs `e-means-4096`)

3-needle haystack (passkeys at 10%, 50% and 90% depth), Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core GPU, 585 GB/s), macOS 26+, `Qwegn v0.14.1`, `--kv-mode q4 --temperature 0 --no-profile`.

| Context rung | Dense (`off`) Prefill / Decode | LSA `Ward-512` Prefill / Decode | **LSA `e-means-4096` Prefill / Decode** | Needle recall | `e-means` gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 64K | 672 / 42.1 | 884.8 / 54.0 | **914.3 / 56.2** | **3/3** | **+3.3% pre · +4.1% dec vs `Ward` (`1.36×` vs Dense)** |
| 160K | 467.4 / 25.9 | 831.9 / **47.1** | **907.0** / 40.7 | **3/3** | **+9.0% prefill vs `Ward` (`1.94×` vs Dense)** |
| 262K | 351.0 / 25.3 | 830.2 / **43.1** | **896.5** / 42.1 | **3/3** | **+8.0% prefill vs `Ward` (`2.55×` vs Dense)** |
| 524K | — | 831.2 / **41.1** | **891.0** / 36.1 | **3/3** | **+7.2% prefill (`-41.8 s` TTFT)** |
| 1M | — | 804.6 / **36.5** | **830.6** / 35.6 | **3/3** | **+3.2% prefill (`-40.5 s` TTFT · `-270 MiB` RSS)** |

**Context labels are nominal benchmark rungs**, matching the infographic. Actual tokenized prompt lengths: 64,060 (64K), 120,673 (128K), 245,423 (262K), 544,656 (524K), and approximately 1,037,119 (1M; SenseNova: 1,037,138). Not every model was measured at every rung. Decode uses native MTP and adaptive long-context backoff, not the serial baseline above. The 3/3 result is one three-needle retrieval exercise, not a general accuracy guarantee. Extending beyond the upstream 262,144-token context is an inference-time extension, not additional training.

LSA restricts each full-attention query to the sink tokens, the top-*k* most similar 4-token blocks (k = 2048 tokens) and a 512-token local window. These retrieval measurements improve alongside the reduced attention workload; they do not establish fidelity on every task.

### LSA sidecar (`model.qlsa`)

| Field | Value |
| :--- | :--- |
| Format | `QLSA0001`, offline SVD of the quantized `wq`/`wk` RoPE-free head slice (dims 64–255) |
| Index dimension | 64 |
| Clustering | 512-token superblocks → 64 Ward buckets (≤ 16 tokens each), 4-token key pooling |
| Default budget | 2,048 tokens + sink block + 512-token local window |
| Layers covered | 16 physical full-attention layers; 16 runtime occurrences |
| Dense bypass | prompts under 2,564 tokens run bit-exact dense attention |

Without the sidecar, `--lsa-mode auto` can synthesize projections from the weights at load time; the sidecar avoids recomputation and pins the supplied projection tensors.

## Quantization: `HQ4_G32_M5`

**5.69 bits per parameter overall.** Quantized linear layers use `HQ4_G32_M5`: each group of 32 weights is rotated by an orthonormal Walsh–Hadamard transform, then stored as 4-bit codes with an FP16 scale and bias (20 bytes per 32 weights = **5 bits per weight**).

| Tensor class | Share of parameters |
| :--- | :---: |
| `HQ4_G32_M5` — all linear layers incl. `lm_head` | 93.8% |
| `BF16` — token embedding, vision tower | 6.2% |

**Why this format**

- **The rotation spreads outliers.** A Hadamard rotation mixes each group so one large weight no longer dictates the scale of the other 31. That keeps 4-bit error low without any calibration data.
- **The rotation is fused into the kernels.** Qwegn applies the fast Walsh–Hadamard transform to activations inside GEMV/GEMM, avoiding a separate unpacked weight buffer. The arithmetic is not literally free.
- **BF16 stays only where it costs little bandwidth.** The token embedding is a one-row gather per token and the vision tower runs once per image, so keeping them in BF16 protects quality without slowing decode.

**Speed impact on Qwegn**

Single-stream decode on Apple Silicon is memory-bandwidth-bound: every generated token streams the active weights once. Shrinking those bytes raises the speed ceiling almost proportionally.

|  | HQ4_G32_M5 | BF16 (same model) |
| :--- | :---: | :---: |
| Weights read per token (est.) | 16.02 GB | 51.25 GB |
| Decode ceiling at 585 GB/s | 36.5 | 11.4 |
| Measured serial decode on Qwegn | **31.2 tok/s** | — |

`HQ4_G32_M5` reads approximately **3.2× fewer active-weight bytes per token** than BF16. These are analytical bandwidth ceilings, **not measured BF16 speedups**. Attention, routing, activation transforms, KV traffic and speculative verification add work. No paired BF16 quality/speed evaluation is reported here. Embeddings are excluded from the bytes-per-token figures (one row gathered per token).

## Memory footprint

Weights are memory-mapped (18.40 GiB on disk). Only full-attention layers keep a KV cache; Gated-DeltaNet layers carry a fixed-size state. Reported process footprints from the measured q4-KV needle runs:

| Context rung | LSA / native QSA | dense |
| :--- | :---: | :---: |
| 64K | 20.16 GiB | 20.04 GiB |
| 262K | 23.86 GiB | 23.59 GiB |
| 1M | 38.36 GiB | not run |

These process-footprint snapshots are **not a minimum RAM requirement** or the full mapped file size; resident pages depend on the prompt and operating system.

LSA retains the full KV cache and adds a pooled-key/cluster index and selection scratch. At index dimension 64 the long-lived index is about **48.625 bytes per token per attention occurrence**, before shared scratch, projection matrices, ring buffers and alignment. The selling point is prefill/decode speed and the measured retrieval result, **not KV memory compression**. Steady-state decode measurements recorded no heap or Metal-buffer allocations.


## Quick start

Install Qwegn from [https://github.com/qwegnai/qwegn](https://github.com/qwegnai/qwegn), then:

```bash
qwg pull qwegnai/Qwen3.8-27B-hq4g32

# OpenAI-compatible server, up to 1M tokens with LSA (--no-load-vision saves idle ViT RAM)
qwg serve ~/.qwg/models/Qwen3.8-27B-hq4g32 \
  --context-tokens 1048576 --kv-mode q4 --lsa-mode auto --no-load-vision
```

Or download with the Hugging Face CLI: `hf download qwegnai/Qwen3.8-27B-hq4g32 --local-dir ~/.qwg/models/Qwen3.8-27B-hq4g32`.

## Requirements

- macOS 26+ on Apple Silicon; M5-class GPUs (Apple10) enable the NAX cooperative-tensor prefill path
- Unified memory for the 18.40 GiB mapped pack, runtime buffers and KV cache; all reported measurements used a 128 GB Mac
- Qwegn 0.8.0 or newer

## Files

| File | Contents |
| :--- | :--- |
| `model.qwp` | Weights in Qwegn's QWP1 container (tensor index at the end of the file) |
| `model.qwp.sha256` | SHA-256 of `model.qwp` |
| `model.qlsa` | LSA index projections |
| `README.md` | This model card |
| `media/` | Per-model Qwegn infographic (SVG + PNG) |
| `LICENSE` | Upstream model license |
| `assets/` | Upstream config, tokenizer, chat template, README and LICENSE; `manifest.json` lists their SHA-256 |

## License and attribution

Apache-2.0, inherited from the upstream model. Weights are derived from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwegn's quantizer; all credit for the model belongs to its original authors. Measurements were taken by the Qwegn team on the hardware listed above and will vary with chip, memory and thermals.
