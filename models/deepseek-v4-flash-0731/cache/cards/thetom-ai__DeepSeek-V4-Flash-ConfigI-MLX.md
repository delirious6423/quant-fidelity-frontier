---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
language:
  - en
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
pipeline_tag: text-generation
library_name: mlx
tags:
  - mlx
  - turboquant
  - config-i
  - moe
  - apple-silicon
quantized_by: thetom-ai
inference: false
---

# DeepSeek-V4-Flash — Config-I (MLX)

**284B-parameter MoE (21B active) in 101 GiB — built to fit and run on 128 GB Apple Silicon.** Config-I mixed-precision quantization at **3.05 bpw effective**, standard MLX format.

Config-I quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — the official 2026-07-31 release, superseding the preview — (43 layers, 256 routed experts top-6, hash routing on the first 3 layers, MLA attention, mHC hyper-connections, DSA indexer). The policy applies aggressive 2-bit compression to expert MLPs (where MoE is most tolerant), protects attention at 4-bit, and shields boundary layers, routing, and embeddings at higher precision. See the [Config-I policy writeup](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md) for the derivation.

## Compression

| | Size |
|---|---|
| FP8/FP4 source | 167 GB |
| BF16 equivalent | ~570 GB |
| **Config-I (3.05 bpw)** | **108 GB (101 GiB)** |

## Policy

| Tensor group | Precision |
|---|---|
| Expert gate/up (routed + shared) | 2-bit |
| Expert down | 3-bit |
| MLA attention + indexer + compressor | 4-bit |
| Boundary layers (first 2 + last 2): attention | 8-bit |
| Boundary layers: experts | 4-bit |
| Embeddings + head | 8-bit |
| Router gates, norms, mHC params | f16 |

Group size 64 throughout. The boundary-expert 4-bit choice (vs 8-bit in earlier Config-I releases) is what brings the artifact under the 128 GB unified-memory ceiling with headroom for KV cache and activations.

## Status and testing

> **Note:** this MLX build has had lighter testing than the [GGUF sibling](https://huggingface.co/thetom-ai/DeepSeek-V4-Flash-ConfigI-GGUF). If you want the more thoroughly measured artifact (perplexity, backend traps, behavioral notes), start there.

📖 **How to run it well:** the [offlabel operating guide](https://github.com/TheTom/offlabel/blob/main/models/deepseek-v4-flash-0731.md) covers sampling, failure modes, and the fixes (GGUF-focused, but the sampling and model-behavior notes carry over).

- **Verified:** loads and generates coherently via python mlx-lm (greedy smoke: correctly completes "The opening line of Moby Dick is" and continues cleanly). The model class and quant config load without changes.
- **MLX-specific quality numbers are pending.** This artifact is a distinct quantization from the GGUF sibling (3.05 bpw affine here vs 2.88 bpw TQ hybrid there), so the sibling's perplexity does not transfer and is not quoted here. A full MLX-side PPL / MMLU / NIAH pass is queued; numbers land here when done.
- **Sibling GGUF** (for CUDA / DGX Spark / non-MLX): [DeepSeek-V4-Flash-ConfigI-GGUF](https://huggingface.co/thetom-ai/DeepSeek-V4-Flash-ConfigI-GGUF), which carries measured perplexity, backend notes, and observed behaviors. Note: the fast DSv4 engine on Apple Silicon (DwarfStar / antirez ds4, ~5x decode) runs the GGUF/ds4 format, not MLX; if raw speed matters more than the MLX toolchain, use that path.

## Using it (sampling that matters)

- **Sampling floor:** DeepSeek's official settings are **temperature 1.0, top_p 0.95** (top_p 1.0 outside agentic use). Greedy decoding has a documented looping tendency on this model family; do not evaluate or run it greedy.
- **Give it a real output budget.** Reasoning is verbose by design (the vendor allows very long outputs at higher reasoning effort); a small `max_tokens` truncates mid-thought and looks like a failure.
- **Observed on the sibling quant (may transfer, unverified on this one):** long unattended symbolic-math derivations can fall into repetition; anti-repetition sampling fixes it there. Route hard math with supervision until MLX-side confirmation.

## Running it

- **mlx-swift-lm**: `deepseek_v4` is a registered model type — loads directly.
- **python mlx-lm**: requires the `deepseek_v4` model class (upstream PR pending). Until it lands, drop the included `deepseek_v4.py` into `mlx_lm/models/`.

Peak memory during generation: ~109 GB. On a 128 GB Mac, close other memory-heavy apps; raising `iogpu.wired_limit_mb` helps sustained throughput.

## License

Inherits the DeepSeek model license from the base model. Quantization by [thetom-ai](https://huggingface.co/thetom-ai).
