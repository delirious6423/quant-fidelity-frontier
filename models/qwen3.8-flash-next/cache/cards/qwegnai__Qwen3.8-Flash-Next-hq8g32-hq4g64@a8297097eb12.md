---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: qwegn
pipeline_tag: text-generation
tags:
  - qwegn
  - apple-silicon
  - metal
  - quantized
  - hq4
  - moe
  - hq8
  - latent-sparse-attention
  - long-context
---

# Qwen3.8-Flash-Next · `HQ8_G32 + HQ4_G64` · Qwegn native pack

A native weight pack for the **[Qwegn inference engine](https://github.com/qwegnai/qwegine)**, a Swift + Metal 4 runtime for Apple Silicon. Mixed **`HQ8_G32` dense trunk + `HQ4_G64` routed experts** cuts logit KL divergence vs BF16 by **2.62×** (and **3.56× on nested JSON**) while shrinking the pack by **5.16 GiB** (`71.30 GiB`), delivering **100% conformance, 100% capability, 100% fidelity**, and **3/3 needle recall at 1M tokens**.

<p align="center"><img src="media/qwegn_lsa_scaling.png" alt="Qwen3.8-Flash-Next on Qwegn" width="960"></p>

> This pack uses Qwegn's own `QWP1` container and loads only in Qwegn. It is not a safetensors, GGUF or MLX checkpoint.

## At a glance

| Property | Value |
| :--- | :--- |
| Upstream model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) @ `7ecef55a` |
| Parameters | 128.80 B |
| Architecture | 48-layer hybrid MoE: 36 Gated-DeltaNet + 12 trained sparse-attention (QSA) layers, 512 experts (10 routed + shared), per-layer n-gram embeddings (PLE), MTP head, vision tower |
| Pack size | 71.30 GiB (+ 50.86 GiB PLE n-gram table) |
| Weights | `HQ8_G32 + HQ4_G64` + BF16, 4.76 bits/param |
| Native context | 262,144 tokens; inference extension tested near 1M with native QSA |
| LSA / Sparse Attention | ✅ trained native QSA + `model.qlsa` for `qsa-cluster` |

## Performance on Qwegn

Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core GPU, 585 GB/s), macOS 26; `llmprobe v0.6.16`, `--kv-mode q4 --temperature 0`.

| Metric | Result |
| :--- | :---: |
| Prefill (10.4K prompt) | 2,460–2,688 tok/s (2,696 peak) |
| Cold TTFT | 113–122 ms |
| Warm prompt-cache TTFT | 7–10 ms |
| Standard decode (serial) | 72.3 tok/s |
| Speculative decode (Variational MTP-3) | 106.7–113.7 tok/s (142.4 on predictable text) |
| 1M-token prefill / decode (native QSA) | 1,353 / 45.4 tok/s |
| Needle recall 64K → 1M | 3 / 3 at every length |
| Engine conformance & capability | 100% (131/131) · 100% (18/18, 6/6 JSON) |
| Logit fidelity & agentic tool use | 100% (meanTopProb 0.9658) · 87.5% (7/8) |
| Code eval (50 HumanEval + 50 MBPP) | 81–83 / 100 |

Short-context results are from separate workloads; warm-cache TTFT and the listed cold TTFT are not a controlled same-prompt ratio. Speculative short-context runs use Variational MTP-3 (margin 2.5, probability floor 0.2, hybrid drafting), NextN MTP-1 for `GLM-5.3-Flash`, or DFlash 2 for the drafter row. Variational acceptance can change output versus exact greedy decoding; NextN MTP-1 greedy verification is bit-exact. Code evaluation, where shown, uses 50 HumanEval + 50 MBPP tasks with reasoning disabled. Results vary with workload and thermals.

## Long context with Latent Sparse Attention (LSA)

3-needle haystack (passkeys at 10%, 50% and 90% depth), Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core GPU, 585 GB/s), macOS 26, `llmprobe v0.6.16`, `--kv-mode q4 --temperature 0`. Each cell is Native QSA (`auto`) / `qsa-cluster`.

| Context rung | Prefill tok/s | TTFT | Decode tok/s | Needle recall | QSA vs qsa-cluster |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 64K | 2,542 / 2,280 | 25.2 s / 28.1 s | 81.7 / 63.0 | 3/3 vs 3/3 | 1.11× · 1.30× |
| 262K | 2,171 / 1,553 | 113.1 s / 158.0 s | 72.3 / 34.8 | 3/3 vs 3/3 | 1.40× · 2.08× |
| 1M | 1,353 / — | 766.5 s / — | 45.4 / — | 3/3 | — |

**Context labels are nominal benchmark rungs**, matching the infographic. Actual tokenized prompt lengths: 64,060 (64K), 120,673 (128K), 245,423 (262K), 544,656 (524K), and approximately 1,037,119 (1M; SenseNova: 1,037,138). Not every model was measured at every rung. Decode uses native MTP and adaptive long-context backoff, not the serial baseline above. The 3/3 result is one three-needle retrieval exercise, not a general accuracy guarantee. Extending beyond the upstream 262,144-token context is an inference-time extension, not additional training.

Flash-Next was trained with its own sparse-attention indexer (QSA). `--lsa-mode auto` keeps that trained indexer on Qwegn's paged sparse kernels; the included `model.qlsa` sidecar powers the opt-in `--lsa-mode qsa-cluster` path, which the table compares against.

### LSA sidecar (`model.qlsa`)

| Field | Value |
| :--- | :--- |
| Format | `QLSA0001`, offline SVD of the quantized `wq`/`wk` RoPE-free head slice (dims 64–255) |
| Index dimension | 64 |
| Clustering | 512-token superblocks → 64 Ward buckets (≤ 16 tokens each), 4-token key pooling |
| Default budget | 2,048 tokens + sink block + 512-token local window |
| Layers covered | 12 physical full-attention layers; 12 runtime occurrences |
| Dense bypass | prompts under 2,564 tokens run bit-exact dense attention |

Without the sidecar, `--lsa-mode auto` can synthesize projections from the weights at load time; the sidecar avoids recomputation and pins the supplied projection tensors.

## Quantization: `HQ8_G32 + HQ4_G64` (`dense_hq8_g32__expert_hq4_g64`)

**4.76 bits per parameter overall** (`71.30 GiB` for `128.80 B` parameters). Routed MoE experts (`95.7%` of parameters) use **`HQ4_G64`** (Hadamard-rotated 4-bit codes over 64-weight blocks with FP16 scale and bias = **36 bytes per 64 weights = 4.5 bits/weight**), while the shared trunk (`3.4%` of parameters: QSA attention, Gated-DeltaNet, shared experts, `lm_head`, and MTP dense projections) uses **`HQ8_G32`** (**36 bytes per 32 weights = 9.0 bits/weight**).

| Tensor class | Share of parameters |
| :--- | :---: |
| `HQ4_G64` — routed MoE experts (512 × 48 layers + MTP, 4.5 bits/weight) | 95.7% |
| `HQ8_G32` — QSA, Gated-DeltaNet, shared experts, `lm_head` (9.0 bits/weight) | 3.4% |
| `BF16` — token embedding, vision tower, 4-branch HC & router gates | 0.9% |

**Why this mixed format**

- **2.62× lower KL divergence vs BF16 while 5.16 GiB smaller than uniform `HQ4_G32`.** Upgrading the 3.4% shared trunk to `HQ8_G32` eliminates the single largest quantization bottleneck (cutting overall KLD vs BF16 from `0.02116` to **`0.00807`** and nested-JSON KLD by **3.56×** from `0.01575` to **`0.00442`**), while switching the 512 routed experts per layer to `HQ4_G64` halves group scale/bias overhead and shrinks `model.qwp` from `76.46 GiB` to **`71.30 GiB`**.
- **100% capability & structured JSON discipline.** Eliminates bracket-depth drift on deeply nested JSON schemas (`6/6` on `llmprobe` `json-discipline`, `18/18` capability, `100%` logit fidelity with `meanTopProb = 0.9658`).
- **Fused Metal 4 & Apple10 NAX execution.** Both `HQ8_G32` and `HQ4_G64` execute directly inside Qwegn's fused Hadamard GEMV, tiled NAX GEMM, and row-paired speculative verify (`qwg_aq_slice_dot2`) kernels without weight unpacking.
- **BF16 stays where it costs little bandwidth.** The token embedding is a one-row gather per token, the vision tower runs once per image and the router gates are tiny but decide which experts run, so BF16 protects routing without slowing decode.

**Speed impact on Qwegn**

Single-stream decode on Apple Silicon is memory-bandwidth-bound: every generated token streams the active weights once. Shrinking those bytes raises the speed ceiling almost proportionally.

|  | HQ8_G32 + HQ4_G64 | BF16 (same model) |
| :--- | :---: | :---: |
| Weights read per token (est.) | 4.41 GB | 13.34 GB |
| Decode ceiling at 585 GB/s | 133 | 44 |
| Measured serial decode on Qwegn | **72.3 tok/s** | — |

`HQ8_G32 + HQ4_G64` reads approximately **3.0× fewer active-weight bytes per token** than BF16. These are analytical bandwidth ceilings, **not measured BF16 speedups**. Attention, routing, activation transforms, KV traffic and speculative verification add work. No paired BF16 quality/speed evaluation is reported here. Embeddings are excluded from the bytes-per-token figures (one row gathered per token).
For MoE packs these figures count only the routed experts actually read per token.

## Memory footprint

Weights are memory-mapped (71.30 GiB (+ 50.86 GiB PLE n-gram table) on disk). Only full-attention layers keep a KV cache; Gated-DeltaNet layers carry a fixed-size state. Reported process footprints from the measured q4-KV needle runs:

| Context rung | LSA / native QSA | qsa-cluster |
| :--- | :---: | :---: |
| 64K | 15.58 GiB | 15.62 GiB |
| 262K | 17.74 GiB | 17.81 GiB |
| 1M | 26.31 GiB | not run |

These process-footprint snapshots are **not a minimum RAM requirement** or the full mapped file size; resident pages depend on the prompt, experts touched and operating system. Flash-Next has another recorded 1M run at 26.82 GiB (the infographic rounds that to 26.8 GiB).

LSA retains the full KV cache and adds a pooled-key/cluster index and selection scratch. At index dimension 64 the long-lived index is about **48.625 bytes per token per attention occurrence**, before shared scratch, projection matrices, ring buffers and alignment. The selling point is prefill/decode speed and the measured retrieval result, **not KV memory compression**. Steady-state decode measurements recorded no heap or Metal-buffer allocations.

The LSA index estimate above applies to the opt-in `qsa-cluster` path, not to Flash-Next's default native QSA.
## Quick start

Install Qwegn from [https://github.com/qwegnai/qwegine](https://github.com/qwegnai/qwegine), then:

```bash
qwg pull qwegnai/Qwen3.8-Flash-Next-hq8g32-hq4g64

# OpenAI-compatible server, up to 1M tokens with LSA
qwg serve ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64 \
  --context-tokens 1048576 --kv-mode q4 --lsa-mode auto
```

Or download with the Hugging Face CLI: `hf download qwegnai/Qwen3.8-Flash-Next-hq8g32-hq4g64 --local-dir ~/.qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64`.

## Requirements

- macOS 26+ on Apple Silicon; M5-class GPUs (Apple10) enable the NAX cooperative-tensor prefill path
- Unified memory for the 71.30 GiB mapped pack, runtime buffers and KV cache; all reported measurements used a 128 GB Mac
- Qwegn 0.9.0 or newer

## Files

| File | Contents |
| :--- | :--- |
| `model.qwp` | Weights in Qwegn's QWP1 container (tensor index at the end of the file) |
| `model.qwp.sha256` | SHA-256 of `model.qwp` |
| `model.qlsa` | LSA index projections |
| `README.md` | This model card |
| `media/` | Per-model Qwegn infographic (SVG + PNG) |
| `LICENSE` | Upstream model license |
| `assets/` | Upstream config, tokenizer, chat template, README and LICENSE, PLE n-gram table (`ngram.bin`); `manifest.json` lists their SHA-256 |

## License and attribution

Qwen Community License 1.0 (see `LICENSE`), inherited from the upstream model, including its terms for model-as-a-service use. Weights are derived from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwegn's quantizer; all credit for the model belongs to its original authors. Measurements were taken by the Qwegn team on the hardware listed above and will vary with chip, memory and thermals.
