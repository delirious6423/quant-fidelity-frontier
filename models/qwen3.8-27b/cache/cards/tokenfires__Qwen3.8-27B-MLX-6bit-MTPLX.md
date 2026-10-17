---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
language:
- en
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: mlx
tags:
- mlx
- mtplx
- qwen3_5
- speculative-decoding
- multi-token-prediction
- apple-silicon
- 6-bit
---

# Qwen3.8-27B — MLX 6-bit + MTPLX MTP adapter

A 6-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with a trained multi-token-prediction (MTP) sidecar for [MTPLX](https://github.com/youssofal/MTPLX) — native speculative decoding on Apple Silicon with no external draft model.

Built with MTPLX Forge v2.9.2, which converts, calibrates, and then **verifies the speedup on real hardware** before accepting the adapter. On the build machine (Apple Silicon M5 Max, macOS 26.6.2), Forge measured **2.87× faster decode than autoregressive baseline** at MTP depth 3.

## Contents

| File | Purpose |
|---|---|
| `model-0000*-of-00005.safetensors` | Language model trunk, 6-bit affine quantization (group size 64) |
| `mtp.safetensors` | MTP draft heads, kept BF16 (depth up to 3) |
| `model-vision.safetensors` | Vision tower (333 tensors), included from the base model |
| `mtplx_runtime.json` | Forge provenance, MTP contract, and verified speed evidence |

Total footprint: ~22 GB on disk, ~25.4 GB resident during inference.

## Quantization recipe

- **Trunk:** 6-bit, affine mode, group size 64 (MLX)
- **MTP sidecar:** BF16 (unquantized — draft heads must track the trunk's distribution closely; quantizing them costs acceptance rate)
- **Source:** `Qwen/Qwen3.8-27B` @ [`1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/commit/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), BF16 native

## Measured performance

Forge verification (greedy continuation, single prompt, M5 Max — recorded in `mtplx_runtime.json`):

| MTP depth | Acceptance by position | tok/s | vs. baseline |
|---|---|---|---|
| 0 (autoregressive) | — | 22.6 | 1.00× |
| 1 | 97% | 42.3 | 1.87× |
| 2 | 98% / 94% | 57.4 | 2.53× |
| 3 | 99% / 95% / 89% | **65.1** | **2.87×** |

Verdict: `mtp_depth_wins` — depth 3 fastest on this hardware, quality checks passed at every depth, no configurations rejected.

![Forge verification: depth 3 is fastest on this Mac, 22.6 to 65.1 tok/s, 2.87x faster](assets/forge-verify.png)

Interactive chat (sampled at temperature 0.6, best of 3 runs, same machine):

| Metric | Value |
|---|---|
| Prefill | 60 tok/s |
| Decode | 50 tok/s |
| Draft acceptance | 75% |
| Memory | 25.4 GB |
| Cold-start first prompt | 36.9 tok/s |

Sampled decoding accepts fewer drafts than the greedy verification pass, so real-chat throughput sits below the verify numbers. Both are honest measurements of the same artifact.

![Cold-start chat in MTPLX: first prompt streaming at 36.9 tok/s](assets/chat-cold-start.png)

## Usage

Requires MTPLX ≥ 2.9 on Apple Silicon:

```bash
mtplx pull tokenfires/Qwen3.8-27B-MLX-6bit-MTPLX
mtplx run tokenfires/Qwen3.8-27B-MLX-6bit-MTPLX
```

Recommended sampler settings (shipped in `mtplx_runtime.json`): temperature 0.6, top-k 20, top-p 0.95. Recommended profile: `sustained`.

The trunk is standard MLX 6-bit and also loads in `mlx-lm` without the MTP speedup; the `mtp.safetensors` sidecar is only used by MTPLX.

## Provenance

- Forged locally 2026-08-25 with MTPLX Forge v2.9.2 (`arch_id: qwen3-next-mtp`)
- Full recipe, MTP contract, acceptance data, and artifact fingerprint in `mtplx_runtime.json`
- Base model license: [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) (included as `LICENSE`)

This is a community quantization, not affiliated with Qwen or the MTPLX project.
