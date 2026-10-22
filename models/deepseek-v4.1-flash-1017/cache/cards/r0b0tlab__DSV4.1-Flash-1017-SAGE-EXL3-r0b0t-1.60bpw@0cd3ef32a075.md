---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4.1-Flash-1017
  - vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - deepseek
  - moe
  - quantization
  - mixed-precision
  - r0b0tlab
---

# DeepSeek-V4.1-Flash-1017 — SAGE EXL3 r0b0t 1.60 bpw

Mixed-precision EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), built directly upon [vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw) by **r0b0tlab** (`@mr_r0b0t`).

## Credits & Acknowledgments

Base mixed-precision **SAGE** EXL3 quantization by our collaborator **[vcruz305](https://huggingface.co/vcruz305)** ([`vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw)). Special thanks to **vcruz305** for their excellent work designing and producing the SAGE mixed-precision expert quantization, preserving the in-pack **DSpark** speculative drafter (`mtp.0..2`), unquantized BF16 vision tower, and FP8 Engram tables.

This `r0b0t-1.60bpw` pack extends `vcruz305`'s SAGE pack with activation-aware `K=4` (`4-bit`) EXL3 quantization of `head`, trunk MLA `wq_b` / `wo_b`, and `shared_experts.{w1,w2,w3}`, plus `K=3` (`3-bit`) EXL3 quantization of the `924` outlier routed expert projections in `layers.15..32`, reducing the GPU-resident weight footprint so **100% of all 15,360 trunk routed experts and 384 DSpark drafter experts fit resident in GPU memory on a single 128 GB unified-memory node**.

## Pack Summary & Calculated Bitwidth (`bpw`)

| Item | Value |
| --- | --- |
| Format | EXL3 (`mul1` trellis) + FP8 (`E4M3` + `E8M0` scales) + BF16 vision tower |
| Routed Expert Profile (Trunk + DSpark, incl. `suh`/`svh`/`mul1`) | **`1.60 bpw`** (`1.6018 bpw`; `1.5979 bpw` trunk-only; `1.5878 bpw` raw trellis `K`) |
| All EXL3 Tensors (Expert + Trunk + Head, incl. scales) | **`1.63 bpw`** (`1.6251 bpw`; `1.6150 bpw` raw trellis `K`) |
| All Trunk + Head Linear Weights (EXL3 + FP8, incl. scales) | **`1.64 bpw`** (`1.6420 bpw` across `550.77B` linear parameters) |
| GPU-Resident Weight Footprint (Single-Rank) | **`~110.6 GiB`** (`6.69 GiB` trunk + `101.11 GiB` `15,360/15,360` trunk experts + `2.78 GiB` `384/384` DSpark experts) |
| Total Pack Size | **`300.33 GiB`** (`17` shards `model-00001-of-00017.safetensors` .. `model-00017-of-00017.safetensors`, `190,952` tensors) |
| Index | `model.safetensors.index.json` (SHA-256: `5cd09e9f0107d3036d1600d0524b88b0c94c5df24d4e06d41dda1424ea7fcc22`) |

### Layer Precision Layout

- **`head` (`[5120, 129280]`)**: `4-bit` EXL3 (`K=4`, `mul1`, calibrated with `H = diag(norm.weight^2)`)
- **Trunk MLA `wq_b` (`[1280, 32768]`), `wo_b` (`[8192, 5120]`), and `shared_experts.{w1,w2,w3}`**: `4-bit` EXL3 (`K=4`, `mul1`)
- **Trunk MLA Latent Projections (`wq_a`, `wkv`, `wo_a`)**: `8-bit` FP8 (`E4M3` + `E8M0` scale blocks)
- **Trunk Routed Experts (`40` layers × `384` experts = `15,360` experts)**: `1–3 bit` SAGE EXL3 (`K=1..3`, `mul1`)
- **DSpark Speculative Drafter (`mtp.0..2`)**: In-pack DSpark (`FP8` attention/shared + `EXL3` routed experts)
- **Vision & Engram**: Unquantized `BF16` vision tower (`266` tensors) and host-mapped `FP8` Engram tables (`shards 16–17` + `engram_token_map.u32`)

## Measured Performance (GB10 / SM121)

### Decode Throughput (DSpark Speculative Decoding, `Q4` KV Cache)

| Topology | Single-Stream (`l1_192` Median / Peak) | 8-Prompt Diverse Suite (Median / Code Max) | Concurrent `c=2` (Aggregate) | Concurrent `c=4` (Aggregate) | Resident Routed Experts |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Single-Rank (1× GB10 128 GB)** | **`50.4 tok/s`** (`50.6 tok/s`) | **`27.4 tok/s`** (`50.3 tok/s`) | **`49.5 tok/s`** (`51.6 tok/s`) | **`45.5 tok/s`** (`50.6 tok/s`) | **`15,360 / 15,360` (`100%`)** |
| **TP=2 (2× GB10 over 200 GbE RoCEv2)** | **`59.5 tok/s`** (`61.5 tok/s`) | **`30.0 tok/s`** (`62.5 tok/s`) | **`60.1 tok/s`** (`61.0 tok/s`) | **`49.0 tok/s`** (`52.1 tok/s`) | **`15,360 / 15,360` (`100%`)** |

### Prefill Throughput & Time-to-First-Token (`TTFT`)

| Prompt Length | Single-Rank TTFT (`s`) | Single-Rank Prefill (`tok/s`) | TP=2 TTFT (`s`) | TP=2 Prefill (`tok/s`) |
| ---: | ---: | ---: | ---: | ---: |
| `512` | `0.84 s` | `618 tok/s` | `0.52 s` | `989 tok/s` |
| `2,048` | `1.42 s` | `1,418 tok/s` | `1.16 s` | `1,731 tok/s` |
| `8,192` | `4.53 s` | `1,719 tok/s` | `3.84 s` | `2,028 tok/s` |
| `32,768` | `16.97 s` | `1,823 tok/s` | `14.92 s` | `2,074 tok/s` |
| `131,072` (`128K`) | — | — | `73.56–79.36 s` | `1,638–1,768 tok/s` |

### Long-Context NIAH & Quality Verification

- **128K–1M Long-Context NIAH Passkey & Multi-Key (`10%`, `50%`, `90%`, `33/66/90%` multi-key)**: **`16 / 16` (`100%`) PASS** up to `1,047,521` prompt tokens.
- **Single-Rank vs TP=2 Exactness Gate (`15` prompts + `512..32K` prefill hashes)**: **`100%` byte-identical outputs**.
- **Multimodal & API Capability Matrix (`9/9`)**: **`100%` PASS** (Text, Vision Image, Tool Calling, Reasoning Effort, Multi-Turn Prefix Cache, SSE Streaming, Structured Outputs).

## Measured Base SAGE Quality (`vcruz305`)

Teacher-forced token-level agreement of the base SAGE expert quantization with **DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`), full-vocabulary KL divergence per token, on held-out sequences:

| Category | KL vs official FP8/FP4 release (mean) | KL vs official FP8/FP4 release (`p50`) | Top-1 Agreement | Answer-Only Top-1 | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- |
| **Overall** | **0.0982** | **0.000190** | **94.52%** | — | **81.16%** |
| **Code** | **0.0186** (`0.0046` answer) | `0.000004` | **99.08%** | **99.61%** | **80.38%** |
| **Reasoning** | **0.0349** (`0.0306` answer) | `0.000480` | **96.04%** | **97.98%** | **86.34%** |
| **Math** | **0.0477** (`0.0408` answer) | `0.000306` | **95.56%** | **94.19%** | **83.99%** |
| **General** | **0.2245** | `0.001505` | **89.94%** | **89.21%** | **76.24%** |

## Source Model (DeepSeek)

V4.1-Flash-1017 is DeepSeek's MIT CED MoE: 40 layers, 384 routed experts + 1 shared, top-6. Architecture, sampling defaults (`temperature=1.0`, `top_p=0.95`), and upstream benchmarks are [DeepSeek's](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## License

MIT, same as [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and [vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw). Cite DeepSeek for the base model and vcruz305 for the SAGE quantization.
