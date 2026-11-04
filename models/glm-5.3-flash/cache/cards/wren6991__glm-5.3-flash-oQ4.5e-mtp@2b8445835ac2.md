---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - oq
  - quantized
  - glm5_next
  - mtp
  - 4-bit
  - mixed-precision
---

# GLM-5.3-Flash-oQ4.5e-mtp

A mixed-precision MLX checkpoint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), a natively multimodal mixture-of-experts model with **320B total and 18B active parameters**, 45 layers (42 MoE), 288 routed experts with top-8 routing plus one shared expert, hybrid linear (KDA) + DSA sparse attention, Manifold-Constrained Hyper-Connections (mHC), and a 1M-token context window.

It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

**4.963 bits/weight** over 321.323B logical parameters — **199.33 GB / 185.64 GiB**.

On a held-out KLD sweep it tracks BF16 closer than oQ4e: mean KL 0.124 vs 0.142, top-1 agreement 90.3% vs 89.5% (see [Quality against BF16](#quality-against-bf16)).

> About the name: **oQ4.5e is the label for this layout, not a stock oMLX oQ level.** The build is an oQ4 plan with one extra bit on the routed-expert down projections, so it sits between oQ4e (4.6 bpw) and the 5-bit tier (5.5 bpw). "oQ4.5e" is shorthand; the exact recipe is below and can be reproduced.

## Quantization and Bit Distribution

Built with [oQ](https://github.com/jundot/omlx) (oMLX 0.7.3 release build), mixed precision driven by per-layer sensitivity, importance-matrix rounding (oQe) and a weighted least-squares refit of each group's scale and bias. Group size 64 throughout, affine mode, bfloat16 for scales, biases and unquantized weights.

| Storage format | Logical weights | Share | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 5-bit, group 64 | 103.885B | 32.330% | 66.52 GiB | 5.50 |
| Affine 4-bit, group 64 | 207.886B | 64.697% | 108.91 GiB | 4.50 |
| Affine 8-bit, group 64 | 8.858B | 2.757% | 8.77 GiB | 8.50 |
| BF16 | 0.607B | 0.189% | 1.13 GiB | 16.00 |
| FP32 | 0.086B | 0.027% | 0.32 GiB | 32.00 |
| **Total** | **321.323B** | **100%** | **185.64 GiB** | **4.963** |

The 5-bit bucket is exactly the 43 stacked routed-expert down projections: the 42 MoE layers (3–44) plus the MTP block's MoE layer, each carrying all 288 experts. The 8-bit bucket is 548 modules — every attention projection (KDA and DSA), the DSA indexer, shared experts, the dense MLPs of layers 0–2, `embed_q`/`unembed_out`, `lm_head`, token embeddings and the forget-gate projections. MoE routers, norms, mHC parameters and recurrent (KDA) state stay in FP32/BF16 and are not quantized.

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 313.327B | 180.35 GiB |
| Vision encoder | 0.564B | 1.05 GiB |
| MTP head | 7.433B | 4.24 GiB |
| **Total** | **321.323B** | **185.64 GiB** |

## Compared with Jundot/GLM-5.3-Flash-oQ4e-mtp

The two builds share the same design; the difference is where the extra bits went.

| | Jundot oQ4e | this oQ4.5e |
|---|---|---|
| Size | 186.402 GB / 173.600 GiB | 199.33 GB / 185.64 GiB |
| Effective bits/weight | 4.641 | 4.963 |
| 5-bit weights | — | 103.885B (43 tensors) |
| 4-bit weights | 311.654B | 207.886B |
| 8-bit weights | 8.976B (554 modules) | 8.858B (548 modules) |
| MTP head | 4.013 GiB | 4.24 GiB |
| Vision encoder | 1.050 GiB | 1.05 GiB |

Net cost is **+12.93 GB (+6.9% size)** for one extra bit on every routed expert's down projection. Everything else follows the same design: 4-bit routed `gate_proj`/`up_proj`, 8-bit attention/shared-expert/head/embedding tensors, unquantized routers, norms, mHC and KDA state, BF16 vision tower and a preserved MTP head.

The 8-bit module count differs (548 in this checkpoint vs 554 in oQ4e) because the allocator changed between the two builds: Jundot's checkpoint was made with oMLX 0.7.3.dev1 (PR #4776), this checkpoint with the 0.7.3 release, which includes the refined bit allocation and calibration walk from PR #4930.

## Quality against BF16

Measured with a held-out KLD sweep against the BF16 release, not against another quant.

- **65,536 next-token positions**: 256 sequences × 256-token prefixes.
- **Corpus**: code from public GitHub repositories (CPython, Rust, Go, TypeScript, LLVM, Swift, Linux), plus English, Chinese and Japanese prose from Wikipedia. 8-gram overlap with oMLX's bundled calibration corpora: **0.017%**.
- **Teacher**: BF16 logits computed with a layer-streamed pass over the 642.7 GB checkpoint (one decoder layer resident at a time), with the exact full-vocabulary log-sum-exp per position. KL is evaluated over the teacher's top-256 support, which holds **98.85%** of its probability mass (renormalised).
- **Students**: scored with oMLX's vendored `glm5_next` implementation (see Usage).

| | oQ4e | oQ4.5e |
|---|---|---|
| KL(teacher \|\| student), mean | 0.1420 ± 0.0017 | **0.1244 ± 0.0015** |
| KL, median | 0.0160 | 0.0132 |
| KL, p95 / p99 | 0.638 / 1.803 | 0.550 / 1.613 |
| Top-1 agreement with BF16 | 89.52% | 90.33% |
| Student perplexity | 3.3440 | 3.3330 |
| NLL minus teacher (lower is better) | +0.0283 | +0.0250 |

Teacher perplexity on the same positions is 3.2507. The paired per-position difference is **-0.0177 ± 0.0012** in favour of oQ4.5e (about 15 standard errors), with 58.1% of positions improved; the mean gain is carried by the tail of the distribution, which is what an extra bit on the routed-expert down projections would be expected to do.

Per domain (KL mean, oQ4e → oQ4.5e):

| Domain | oQ4e | oQ4.5e |
|---|---|---|
| Code | 0.1506 | 0.1338 |
| English | 0.1408 | 0.1227 |
| Chinese | 0.1396 | 0.1115 |
| Japanese | 0.1043 | 0.0941 |

Scope: prefix positions only (0-255 of each sequence), a single held-out corpus, and a top-256 rather than full-vocabulary KL. Both builds sit close to BF16 on this sweep; the difference between them is real but small.

## Why the extra bit goes on the expert down projections

At oQ4e, every tensor that writes back into the residual stream is already at 8-bit — `o_proj`, dense `down_proj`, shared-expert `down_proj`, `unembed_out` — *except* the routed experts' down projection, which is 95% of the model's parameters and a third of each expert's weights. This build spends the entire remaining budget on that one tensor type, uniformly across all 288 experts per layer (mixed per-expert widths are not supported by the MoE forward pass).

## How it was built

| | |
|---|---|
| Base checkpoint | `zai-org/GLM-5.3-Flash` BF16 (120 shards, 642.7 GB) |
| Tool | oMLX 0.7.3 (release), `quantize_oq_streaming`, MLX 0.32.4 |
| oQ level | 4 (base 4-bit, target 4.7 / hard cap 4.8 bpw) |
| Extra step | routed `switch_mlp.down_proj` lifted to 5-bit after planning, then re-priced |
| Rounding | oQe: imatrix-weighted rounding + weighted least-squares refit |
| Calibration | `oqe_code_multilingual`, 128 samples × 512 tokens, adaptive coverage |
| Coverage | 38,688 / 38,688 experts seen, p05 = 504 tokens per expert, zero-expert count 0 |
| Sensitivity | static per-layer map from the published oQ4e report (45 layers) |
| Calibration mode | layer-streamed (one decoder layer resident, ~39 GB peak), avoiding oQ's 4-bit calibration proxy |
| Output dtype | bfloat16 |
| Wall time | 19.2 min end-to-end on an M5 Ultra (80-core, 256 GB) |

The sensitivity map came from [`Jundot/GLM-5.3-Flash-oQ4e-mtp/oq_imatrix_report.json`](https://huggingface.co/Jundot/GLM-5.3-Flash-oQ4e-mtp/blob/main/oq_imatrix_report.json), so the layer priorities are Jundot's measurements for that checkpoint rather than a fresh sensitivity pass measured for this checkpoint.

## Usage

oMLX 0.7.3 or newer. `mlx-lm` does **not** support this architecture.

If you use `mlx-vlm` directly: upstream mlx-vlm 0.7.11 and earlier load these weights but produce near-uniform logits for this checkpoint (KL ~10.7, top-1 agreement 0.5%). Use the implementation from [mlx-vlm PR #2030](https://github.com/Blaizzy/mlx-vlm/pull/2030) — oMLX vendors it and exposes it as `apply_mlx_vlm_glm5_next_compat_patch()` — until that lands upstream.

```bash
omlx serve --model-dir /path/to/models --port 8080
```

The MTP head is preserved, so Lightning MTP and DFlash2 speculative decoding work with this checkpoint. Recommended sampling from the upstream GLM-5.3-Flash model card: `temperature: 1.0`, `top_p: 0.95`, `reasoning_effort: max`, thinking enabled, `clear_thinking: false`.

## Memory

185.6 GiB of weights. On a 256 GB Apple-silicon machine that leaves roughly 55–60 GiB for the OS, KV cache and prefix cache — comfortable at moderate context, tight at very long context. The checkpoint has been verified to load and serve on an M5 Ultra with 256 GB of unified memory.

## Caveats

- This is a custom layout produced from the upstream BF16 release; it is not an official oMLX oQ level and has not been validated by the oQ maintainers.
- GLM-5.3-Flash 4-bit builds have an open upstream issue where the model can occasionally sample `<|assistant|>` at tool-call boundaries and run into fabricated dialogue ([jundot/omlx#3271](https://github.com/jundot/omlx/issues/3271)). This checkpoint has not been evaluated specifically for that behaviour; if you hit it in agentic use, suppress token `154828` in the output logits rather than adding it to `eos`.
- Quality is measured only by the 65k-position held-out KLD sweep above; it is not a full benchmark suite, and the sweep covers prefix positions only.

## License

MIT, following the source model.
