---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- quantized
- trellis
- q4t
- apple-silicon
- speculative-decoding
- mtp
- dflash
---

# mavis-ai/Qwen3.8-27B-Q4T

Q4T build of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for **R.E.V.I.S.** on Apple silicon, bundled with its speculative-decoding drafters (DFlash 2 and MTP).

mavis-ai claims no ownership of this model. It is a data-free quantized derivative of the official Qwen3.8-27B checkpoint; all rights in the underlying model remain with its original authors, and it is redistributed under the Apache License 2.0. It is not a fine-tune: no training of any kind was applied. Source revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (read through its BF16 MLX conversion [`mlx-community/Qwen3.8-27B-bf16`](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16), revision `6f265714824f3c38d4452baa1628aef3d9b9aae9`).

> **Compatibility.** The MLP projections are stored as trellis-coded tensors (`.trellis`, `.suh`, `.svh`) and need a special trellis decoding computation. This model therefore currently does **not** run on `mlx-lm` (nor on `mlx-vlm`, `transformers` or vLLM). It runs in **R.E.V.I.S. v1.3.0 and later**, whose built-in inference engine provides the kernels.

## R.E.V.I.S.

This model is made for **R.E.V.I.S.**, a 100% local Cognitive OS for Multi-Agentic AI on Mac. This model is supported from **R.E.V.I.S. v1.3.0**.

- **🌐 Official Website:** [https://mavis-ai.co.jp/revis/](https://mavis-ai.co.jp/revis/)
- **🐦 Follow our updates on X:** [@mavis_ai_jp](https://x.com/mavis_ai_jp)

## Summary

| | |
|---|---|
| Base model | Qwen3.8-27B (multimodal dense model; 64 layers: 48 Gated DeltaNet + 16 full attention, hidden size 5120, MLP intermediate size 17408, vocabulary 248320, context length 262144) |
| Input | text, images, video |
| Format | Q4T: K4 MLP + 6-bit projections |
| MLP (gate / up / down of all 64 layers) | 4-bit trellis-coded (K4) |
| Attention and Gated DeltaNet projections | 6-bit affine, group size 64 (three tensors 8-bit by rule R1) |
| Token embeddings and LM head | 8-bit affine, group size 64 |
| Gated DeltaNet `in_proj_a` / `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, norms | BF16 |
| Vision tower | BF16 |
| Calibration data | none (data-free) |
| Weight size | 18.13 GB (decimal), 16.9 GiB |
| Bundled drafters | DFlash 2 `DFlash2-Q8`, MTP `Draft-Q8` (+ `ProposalHead`) (3.41 GB together) |
| KV cache (runtime) | 8-bit |

## Bundle layout

```text
.
├── README.md, LICENSE, .gitattributes, SHA256SUMS
├── config.json, generation_config.json
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, chat_template.jinja
├── preprocessor_config.json, processor_config.json, video_preprocessor_config.json
├── model-core-00001-of-00001.safetensors          9.56 GB        projections, embeddings, LM head, BF16 tensors, vision
├── model-trellis-0000{1..8}-of-00008.safetensors  8 x 1.07 GB    trellis-coded MLP, 8 layers per file
├── model.safetensors.index.json
├── quantization_policy.json, dense_affine_manifest.json
├── q4t_manifest.json, q4t-split-lineage.json, q4t_sources.json
└── drafts/
    ├── SHA256SUMS
    ├── Qwen3.8-27B-DFlash2-Q8/              DFlash 2 block drafter, 8-bit affine (g64), 2.35 GB
    ├── Qwen3.8-27B-Draft-Q8/                MTP drafter, 8-bit affine (g64), 0.50 GB
    └── Qwen3.8-27B-Draft-Q8-ProposalHead/   proposal head for the MTP drafter, 0.56 GB
```

The core file and the trellis files are split so the runtime can map the MLP tensors independently; the trellis payloads are byte-identical to the encodes they were measured with (`q4t_sources.json`, `q4t-split-lineage.json`). `SHA256SUMS` lists every file of this repository except itself, `drafts/SHA256SUMS` lists the files under `drafts/`, and per-file SHA-256 values for the root files are also recorded in `q4t_manifest.json`. `quantization_policy.json` and `q4t_manifest.json` are the converter's records: their flags (`product_loader`, `product_load_attempted`, `runtime_forward_verified`, `inference_test_run`) describe the conversion step, which verified the stored bytes against the specification, not an engine forward pass; engine support for this pack came afterwards, with R.E.V.I.S. v1.3.0.

## Quantization (Q4T)

The T family (Q5T, Q4T, Q3T) is a fixed, data-free recipe: one width per module class, chosen by a written rule rather than by looking at evaluation data. On this dense model the formats differ in the MLP trellis rate K and the width of the projections. This one is **K4 + 6-bit projections**.

- **MLP, K4.** The gate, up and down projections of all 64 layers (192 tensors) are stored as trellis codes (4 bits per weight, EXL3-style MCG codebook, recipe `v3-gss110`), with a 128-point Hadamard rotation and per-channel sign vectors (`suh`, `svh`). Encoding minimizes rotated-weight MSE; there is no calibration set, Hessian rounding, importance matrix or quantization-aware training. The trellis decoder and its kernels are part of the R.E.V.I.S. inference engine.
- **Projections, 6-bit.** Every Gated DeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj` and every attention `q_proj` / `k_proj` / `v_proj` / `o_proj` is 6-bit affine, group size 64, quantized from the BF16 source with `mx.quantize` (round to nearest).
- **Rule R1, 8-bit.** A tensor whose trellis encode error (relative MSE) exceeds 1.5x the median of its class at any measured rate is stored 8-bit affine at every tier. The rule reads the weights only; it selects the Gated DeltaNet `out_proj` of layers 0 and 1 and the attention `o_proj` of layer 3.
- **Embeddings and LM head, 8-bit.** `embed_tokens` and `lm_head` are 8-bit affine, group size 64.
- **BF16 kept as-is.** The Gated DeltaNet selection projections `in_proj_a` / `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, all norms, and the whole vision tower with its multimodal projection.

| Format | MLP | Attention and Gated DeltaNet projections | Embeddings and LM head |
|---|---|---|---|
| Q5T | K5 | 8-bit affine, g64 | 8-bit affine, g64 |
| Q4T | K4 | 6-bit affine, g64 (R1 tensors 8-bit) | 8-bit affine, g64 |
| Q3T | K3 | 6-bit affine, g64 (R1 tensors 8-bit) | 8-bit affine, g64 |

## Benchmarks

The question these benchmarks answer is how well each quantization suppresses degradation relative to the full-precision model (Full, BF16). All arms are scored on the same inputs against the same BF16 reference. The final numbers will come from a single benchmark run and are not available yet; the table below is intentionally empty.

| Metric | Full (BF16) | Q8 | Q6 | Q5 | Q4 | Q5T | Q4T | Q3T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KLD vs Full (lower is better) |   |   |   |   |   |   |   |   |
| Top-1 agreement with Full |   |   |   |   |   |   |   |   |
| Accuracy benchmarks (to be defined) |   |   |   |   |   |   |   |   |

### Interim figures (to be replaced)

Earlier internal measurement, kept only until the benchmark above is filled in. Teacher-forced next-token distributions of the BF16 Qwen3.8-27B reference and of each arm, computed by a reference forward pass of the stored weights on cloud GPUs (Colab) with an 8-bit KV cache, **not** by the R.E.V.I.S. engine. KLD is the mean KL divergence from the BF16 distribution (lower is better); top-1 is the share of positions where the arm's most likely token equals BF16's. Short context: 48 documents, brackets are 99.8 % confidence intervals for KLD and 99 % for top-1. Long context: 8 books of 32,768 tokens each, with the 16K-32K position bucket in parentheses. "Ordinary MLX" arms are plain MLX affine quantization (round to nearest) at the stated width. These are distribution-distance measurements, not task benchmarks, and they are **interim**: the benchmark table above supersedes them.

| Arm | KLD vs BF16, 48 documents | Top-1 agreement | KLD, 8 books x 32K | Weight size |
|---|---:|---:|---:|---:|
| Q5T | 0.0034 [0.0030, 0.0037] | 0.9713 [0.9681, 0.9747] | 0.0035 (0.0035) | 22.05 GB |
| Ordinary MLX 5-bit | 0.0109 | n/a | n/a | n/a |
| **Q4T (this repository)** | 0.0112 [0.0100, 0.0126] | 0.9511 [0.9470, 0.9550] | 0.0114 (0.0111) | 18.13 GB |
| Q3T | 0.0353 [0.0313, 0.0393] | 0.9199 [0.9112, 0.9291] | 0.0394 (0.0383) | 15.99 GB |
| Ordinary MLX 4-bit | 0.0411 | n/a | n/a | n/a |
| Ordinary MLX 3-bit | 0.1610 | n/a | n/a | n/a |

Paired on the same documents, Q4T's KLD is 0.0298 lower than ordinary MLX 4-bit (interval [-0.0333, -0.0258]) and within noise of ordinary MLX 5-bit (+0.0003, interval [-0.0005, +0.0014]).

## Runtime

- Runtime: the inference engine built into **R.E.V.I.S.**, a native Swift / C++20 / MLX engine with ahead-of-time compiled Metal kernels for Apple silicon. It has no Python, localhost server or just-in-time Metal in the inference path.
- Speculative decoding: two drafters are bundled under `drafts/`, DFlash 2 and MTP (with its `ProposalHead`). The R.E.V.I.S. engine chooses the drafter; DFlash 2 is its drafter mode for Qwen3.8-27B. Each drafter is only valid with this target. The `ProposalHead` is pinned to this pack's header digest; on any other target the runtime ignores it and keeps the exact output head.
- Losslessness: with greedy decoding, output with a drafter is intended to equal plain decoding of this target, up to the rounding ties that the runtime certifies. Drafters are an execution-speed feature of the runtime and do not change the target's output.
- Chips: the runtime targets M1 to M5; development and verification so far were on M4- and M5-family Macs.
- Sampling defaults in `generation_config.json` follow the base model card (`temperature 1.0`, `top_k 20`, `top_p 0.95`).
- Speed depends on the runtime and the machine, not on this model file, and is therefore not reported here.

## Memory

Weights on disk and in memory: 18.13 GB. The three drafters add 3.41 GB if all are loaded. The KV cache is 8-bit and grows with context (the 16 full-attention layers hold K/V for every context token; the 48 Gated DeltaNet layers keep a fixed-size state); the runtime admits a load against the available memory budget and reports the largest context that fits. Budget for weights plus drafters plus KV and transient buffers when choosing a Mac.

## Provenance of the bundled drafters

Everything in this repository ships together, so the origin of every bundled component is listed here. All bundled drafters are derivatives of their upstream releases; mavis-ai claims no ownership of any of them, and each keeps its upstream license (Apache License 2.0).

| Directory | What it is | Original authors and upstream repository | Upstream revision |
|---|---|---|---|
| (root) | the target model, converted to this format | Qwen Team, [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `drafts/Qwen3.8-27B-DFlash2-Q8/` | DFlash 2 block-diffusion drafter, 8-bit affine (group size 64); `fc`, the candidate selector, the dynamic convolution group and norms stay BF16 | Inco AI, [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2); DFlash 2: [blog](https://inco.ai/blog/dflash2/), code [z-lab/dflash](https://github.com/z-lab/dflash); built on DFlash, paper [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| `drafts/Qwen3.8-27B-Draft-Q8/` | MTP drafter: the multi-token-prediction module shipped inside Qwen3.8-27B, extracted as a standalone drafter and quantized to 8-bit affine (group size 64); `fc` and norms stay BF16 | Qwen Team, [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `drafts/Qwen3.8-27B-Draft-Q8-ProposalHead/` | proposal head for the MTP drafter, made by mavis-ai (see below) | derived from the BF16 `lm_head.weight` of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

**ProposalHead (derived artifact).** `proposal_head.safetensors` is a 3-bit affine (group size 64) copy of the base model's BF16 output head (`lm_head.weight`). The runtime uses it only to choose which blocks of 128 token ids the MTP drafter's proposal is searched in; the chosen rows are then rescored through the exact output head of this pack, and the target model verifies every proposal, so it cannot change the output. `proposal_head.json` pins it to the MTP drafter and to this pack (header digests); on any other target the runtime ignores it and keeps the exact head. No training was involved.

The `README.md` inside each drafter directory restates its origin and license. Upstream benchmark material is not reproduced in this repository.

## License and attribution

mavis-ai claims no ownership of this model. This repository redistributes a quantized derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), released by the Qwen Team under the Apache License 2.0. A copy of the license is in `LICENSE`.

**Modification notice:** the official checkpoint was converted to the Q4T layout described above (K4 trellis-coded MLP, 6-bit affine projections with three 8-bit tensors, 8-bit affine embeddings and LM head, BF16 Gated DeltaNet selection tensors, norms and vision tower). No fine-tuning, additional training or architecture change was applied.

The bundled drafters keep their upstream licenses (Apache License 2.0); see [Provenance of the bundled drafters](#provenance-of-the-bundled-drafters).

## Citation

Please cite the base model:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```

and, if you use the DFlash 2 drafter:

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}

@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
