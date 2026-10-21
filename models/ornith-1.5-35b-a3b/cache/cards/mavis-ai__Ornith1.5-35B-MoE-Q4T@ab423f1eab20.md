---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- ornith-1.5
- moe
- quantized
- trellis
- q4t
- apple-silicon
- speculative-decoding
- mtp
- dflash
---

# mavis-ai/Ornith1.5-35B-MoE-Q4T

Q4T build of [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for **R.E.V.I.S.** on Apple silicon, bundled with its speculative-decoding drafters (MTP and DFlash).

mavis-ai claims no ownership of this model. It is a data-free quantized derivative of the official Ornith-1.5-35B-A3B checkpoint; all rights in the underlying model remain with its original authors, and it is redistributed under the license the upstream model card declares (MIT). It is not a fine-tune: no training of any kind was applied. Source revision: `10fbf86fed7ecee4a061f8b499a618f46001cac1`.

> **Compatibility.** The routed experts are stored as trellis-coded tensors (`.trellis`, `.suh`, `.svh`) and need a special trellis decoding computation. This model therefore does **not** run on `mlx-lm` (nor on `mlx-vlm`, `transformers` or vLLM). It is made for **R.E.V.I.S.**, whose built-in inference engine provides the kernels; the R.E.V.I.S. release that supports this Ornith build will be announced.

## R.E.V.I.S.

This model is made for **R.E.V.I.S.**, a 100% local Cognitive OS for Multi-Agentic AI on Mac.

- **🌐 Official Website:** [https://mavis-ai.co.jp/revis/](https://mavis-ai.co.jp/revis/)
- **🐦 Follow our updates on X:** [@mavis_ai_jp](https://x.com/mavis_ai_jp)

## Summary

| | |
|---|---|
| Base model | Ornith-1.5-35B-A3B (multimodal MoE; 40 layers, hidden size 2048, 256 routed experts with 8 active plus a shared expert, context length 262144) |
| Format | Q4T: K4 routed experts + N6 dense |
| Routed experts | 4-bit trellis-coded (K4) |
| Dense / attention / shared-expert / embedding / LM head | 6-bit affine, group size 64 (N6) |
| Routers and selection layers | BF16 |
| Vision tower | BF16 |
| Calibration data | none (data-free) |
| Weight size | 19.18 GB (decimal), 17.9 GiB |
| Bundled drafters | MTP `Ornith1.5-35B-A3B-MTP-Draft-Q8` (+ `ProposalHead`), `Ornith1.5-35B-A3B-DFlash-Q8` (1.57 GB together) |
| KV cache (runtime) | 8-bit |

## Bundle layout

```text
.
├── README.md, .gitattributes, assets/ (upstream images)
├── config.json, configuration.json, generation_config.json, quantization_policy.json
├── tokenizer.json, tokenizer_config.json, vocab.json, chat_template.jinja
├── preprocessor_config.json, processor_config.json, video_preprocessor_config.json
├── model-core-00001-of-00001.safetensors        2.91 GB   dense, attention, embeddings, routers, vision
├── model-experts-0000{1..8}-of-00008.safetensors  8 x 2.03 GB   trellis-coded routed experts
├── model.safetensors.index.json
├── dense_n6_manifest.json, q4t_manifest.json, q4t-split-lineage.json, q4t_sources.json
└── drafts/
    ├── SHA256SUMS
    ├── Ornith1.5-35B-A3B-MTP-Draft-Q8/              MTP drafter (the base checkpoint's MTP module), 8-bit affine (g64)
    ├── Ornith1.5-35B-A3B-MTP-Draft-Q8-ProposalHead/ proposal head for the MTP drafter
    └── Ornith1.5-35B-A3B-DFlash-Q8/   DFlash block drafter, 8-bit affine (g64)
```

The core file and the expert files are split so the runtime can map experts independently; expert tensors are byte-identical to the unsplit source pack (`q4t-split-lineage.json`). Checksums for the files under `drafts/` are in `drafts/SHA256SUMS`; per-file SHA-256 values for the root files are recorded in `q4t_manifest.json` (the converter's record: its `files` table also lists local download-cache entries (`.cache/…`) and a `.gitattributes`, which are not part of this repository (the `.gitattributes` here is the Hub's own), and its flags describe the conversion step, which verified the effective weights, not an engine forward pass). `quantization_policy.json` is carried over from the affine baseline build and identifies the source repository; the quantization actually applied is the one described below and in `q4t_manifest.json`.

## Quantization (Q4T)

The T family (Q5T, Q4T, Q3T) is a fixed, data-free recipe: one width per module class, chosen by a written rule rather than by looking at evaluation data. The formats differ only in the expert rate K and the dense width N. This one is **K4 + N6**.

- **Routed experts, K4.** Each expert matrix is stored as 16x16-tile trellis codes (4 bits per weight, MCG codebook), with a 128-point Hadamard rotation and per-channel sign vectors (`suh`, `svh`). Encoding minimizes rotated-weight MSE; there is no calibration set, Hessian rounding, importance matrix or quantization-aware training. The trellis decoder and its kernels are part of the R.E.V.I.S. inference engine.
- **Everything else linear, N6.** Attention and linear-attention projections, shared expert, token embeddings and LM head are 6-bit affine, group size 64 (N6), quantized from the BF16 source with `mx.quantize` (round to nearest).
- **BF16 kept as-is.** Router (`mlp.gate`), `shared_expert_gate`, linear-attention `in_proj_a`/`in_proj_b`, norms, and the whole vision tower. Expert selection is a discontinuous top-k decision, so these tensors are never quantized.

| Format | Routed experts | Dense |
|---|---|---|
| Q5T | K5 | N8 (8-bit affine, g64) |
| Q4T | K4 | N6 (6-bit affine, g64) |
| Q3T | K3 | N6 (6-bit affine, g64) |

## Benchmarks

**Preliminary (2026-10-05):** next-token distribution distance and draft acceptance; task accuracy follows. Every number below compares an arm with the full-precision model (BF16) on the same inputs. The arms were run on GPU research harnesses that decode this family's stored weight codes exactly, not through the R.E.V.I.S. engine. Speed is not reported.

### Next-token distribution distance (KLD vs BF16)

Teacher-forced next-token distributions of Ornith-1.5-35B-A3B on 48 documents (17,877 scored positions in three strata: WikiText-2, technical documents, and responses generated by Qwen3.6-35B-A3B, the same token sequences as on our Qwen3.6 cards), full vocabulary, KV cache emulated at 8 bits. KLD is the mean KL divergence from the BF16 distribution (lower is better); top-1 is the share of positions where the arm's most likely token equals BF16's. Brackets are 99.8 % bootstrap confidence intervals over documents.

| Arm | KLD vs BF16 [99.8 % CI] | Top-1 agreement | Weight size |
|---|---:|---:|---:|
| BF16 through bf16 matrix multiplies (floor) | 0.0159 [0.0132, 0.0199] | 0.948 | — |
| **Q4T (this repository)** | 0.0414 [0.0351, 0.0490] | 0.916 | 19.18 GB |
| Q3T | 0.0691 [0.0585, 0.0849] | 0.891 | 15.15 GB |
| Q4, official [`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit) | 0.1328 [0.1166, 0.1535] | 0.847 | 19.51 GB (text only) |
| Q5, the same recipe at 5 bits | 0.0563 [0.0472, 0.0669] | 0.898 | ~23.9 GB (estimated, text only) |
| Q6, official [`ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit) | 0.0311 [0.0265, 0.0364] | 0.926 | 28.17 GB (text only) |

- **Ordinary quantization arms.** The official MLX 4-bit and 6-bit builds use the standard mlx-lm recipe (affine, group size 64, routers and shared-expert gates at 8 bits); we checked that every one of the 512 quantized tensors of the official 4-bit build equals round-to-nearest quantization of the BF16 weights. The 5-bit arm applies the same recipe at 5 bits. The official builds contain the text model only; the T builds also carry the BF16 vision tower.
- **Floor.** "BF16 through bf16 matrix multiplies" is the same BF16 weights computed with bf16 instead of FP32 matrix multiplies. It shows how far this model's distribution moves from arithmetic alone: about 2.7 times as far as Qwen3.6-35B-A3B's under the same test (0.0058), and its expert selection changes more (about 5 % of top-8 expert choices, against 3 % for Qwen3.6). Read the quantized arms against this floor.
- **Pairs** (paired over the same positions, 99.8 % CI): Q4T − official Q4: −0.091 [−0.105, −0.079]; Q3T − official Q4: −0.064 [−0.072, −0.056]; Q4T − 5-bit: −0.015 [−0.023, −0.009]; Q3T − 5-bit: +0.013 [+0.003, +0.022]; Q4T − official Q6: +0.010 [+0.008, +0.014].

### Speculative drafting acceptance

Mean accepted drafts per verification round (the bonus token is not counted) for the two bundled drafters, the MTP drafter (the MTP module shipped inside the base checkpoint) and the DFlash drafter, each at its 8-bit weights. 80 fixed prompts, 16 per workload (code, prose, math, Japanese, tool calls), thinking on, sampled with the model's defaults (temperature 1.0), 3 seeds (48 generations per cell), up to 16,384 generated tokens. The target is this repository's weights decoded exactly, run in vLLM 0.30.0 on a cloud GPU (Google Colab) with a BF16 KV cache (not the R.E.V.I.S. engine). Each cell gives widths 3 / 5 / 7 (drafts proposed per round; 3 and 5 are exact truncations of the same rounds) and a 95 % bootstrap interval over prompts for width 7.

| Workload | MTP, bundled | DFlash, bundled |
|---|---:|---:|
| Code | 1.99 / 2.57 / 2.82 [2.70, 2.95] | 1.87 / 2.59 / 3.11 [2.91, 3.31] |
| Prose | 1.35 / 1.61 / 1.71 [1.50, 1.93] | 1.17 / 1.41 / 1.55 [1.34, 1.76] |
| Math | 2.17 / 2.80 / 3.06 [2.89, 3.21] | 2.24 / 3.24 / 3.96 [3.68, 4.25] |
| Japanese | 1.40 / 1.63 / 1.71 [1.50, 1.91] | 1.21 / 1.55 / 1.74 [1.42, 2.10] |
| Tool calls | 2.20 / 3.00 / 3.40 [3.19, 3.62] | 2.11 / 3.04 / 3.72 [3.34, 4.06] |

- **Against the BF16 target** (same drafter, paired by prompt and seed): every workload and width is within ±0.09 drafts per round of BF16, and no 95 % interval excludes zero.
- **Greedy decoding** (temperature 0, 16 generations per cell; DFlash runs greedy-only in R.E.V.I.S.), width 7, code / prose / math / Japanese / tool calls: DFlash 3.53 / 1.81 / 4.22 / 1.89 / 4.13; MTP 3.12 / 2.10 / 3.22 / 1.87 / 3.54.
- **Failures are excluded, not averaged in.** No sampled generation degenerated or looped; 1 greedy DFlash generation (a Japanese math prompt) looped and is excluded. 7 of 640 generations for this target reached the 16,384-token cap with the reasoning unfinished; all of them come from one MATH-500 prompt that does so in nearly every run.
- Accepted drafts per round are not a speed measurement: the time a round takes depends on the runtime and the machine.

## Runtime

- Runtime: the inference engine built into **R.E.V.I.S.**, a native Swift / C++20 / MLX engine with ahead-of-time compiled Metal kernels for Apple silicon. It has no Python, localhost server or just-in-time Metal in the inference path.
- Speculative decoding: **MTP is the default drafter**; **DFlash is optional**. Both are bundled under `drafts/`. Each drafter is only valid with this target. The `ProposalHead` is pinned to this pack's header digest; on any other target the runtime ignores it and keeps the exact output head. The MTP drafter is the MTP module shipped inside the base checkpoint, converted to 8-bit (see [Provenance of the bundled drafters](#provenance-of-the-bundled-drafters)). The runtime proposes 2 drafts per round by default. On the R.E.V.I.S. engine (M4 Pro and M5 Pro Macs, Q4T target, greedy code prompts) 2 drafts per round was the fastest width at 4K tokens of context (3 within 2 %) and 3 at 32K; 4, 5 and 7 drafts per round were slower than 2 at both lengths despite their higher acceptance, because every verification round costs more.
- Losslessness: with greedy decoding, output with a drafter is intended to equal plain decoding of this target, up to the rounding ties that the runtime certifies. Drafters are an execution-speed feature of the runtime and do not change the target's output.
- Chips: the runtime targets M1 to M5; development and verification so far were on M4- and M5-family Macs.
- Sampling defaults in `generation_config.json` follow the base model (`temperature 1.0`, `top_k 20`, `top_p 0.95`).
- Speed depends on the runtime and the machine, not on this model file, and is therefore not reported here.

## Memory

Weights on disk and in memory: 19.18 GB. The bundled drafters add 1.57 GB if all are loaded. The KV cache is 8-bit and grows with context; the runtime admits a load against the available memory budget and reports the largest context that fits. Budget for weights plus drafters plus KV and transient buffers when choosing a Mac.

## Provenance of the bundled drafters

Everything in this repository ships together, so the origin of every bundled component is listed here. All bundled drafters are derivatives of their upstream releases; mavis-ai claims no ownership of any of them, and each keeps its upstream license.

| Directory | What it is | Original authors and upstream repository | Upstream revision |
|---|---|---|---|
| (root) | the target model, converted to this format | Ornith team, [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | `10fbf86fed7ecee4a061f8b499a618f46001cac1` |
| `drafts/Ornith1.5-35B-A3B-MTP-Draft-Q8/` | MTP drafter: the MTP module shipped inside the base checkpoint (its `mtp.*` tensors), converted to the MLX drafter layout and quantized to 8-bit affine (group size 64, round to nearest, no calibration data); `fc`, router, shared-expert gate and norms stay BF16. License: MIT, as declared upstream | Ornith team, [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | `10fbf86fed7ecee4a061f8b499a618f46001cac1` |
| `drafts/Ornith1.5-35B-A3B-MTP-Draft-Q8-ProposalHead/` | proposal head for the MTP drafter, made by mavis-ai (see below) | derived from the BF16 `lm_head.weight` of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | `10fbf86fed7ecee4a061f8b499a618f46001cac1` |
| `drafts/Ornith1.5-35B-A3B-DFlash-Q8/` | DFlash block-diffusion drafter, 8-bit affine (group size 64); `fc` and norms stay BF16. License: MIT, as declared upstream | Ornith team, [`ornith-ai/Ornith-1.5-35B-A3B-DFlash`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash); DFlash method and code: [z-lab/dflash](https://github.com/z-lab/dflash), paper [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | `942d20fee84d04c9ecbf18e9c6fc4144914a1d5d` |

**ProposalHead (derived artifact).** `proposal_head.safetensors` is a 3-bit affine (group size 64) copy of the base model's BF16 output head (`lm_head.weight`). The runtime uses it only to choose which blocks of 128 token ids the MTP drafter's proposal is searched in; the chosen rows are then rescored through the exact output head of this pack, and the target model verifies every proposal, so it cannot change the output. `proposal_head.json` pins it to the MTP drafter and to this pack (header digests); on any other target the runtime ignores it and keeps the exact head. No training was involved.

The `README.md` inside each drafter directory restates its origin and license. The drafters' upstream benchmark material is not reproduced here; `assets/` holds the base model's upstream images unchanged.

## License and attribution

mavis-ai claims no ownership of this model. This repository redistributes a quantized derivative of [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), released by the Ornith team; its model card declares the MIT License (the upstream repository ships no separate license file at the source revision).

**Modification notice:** the official checkpoint was converted to the Q4T layout described above (K4 trellis-coded routed experts, 6-bit affine dense tensors, BF16 routers and vision tower). No fine-tuning, additional training or architecture change was applied.

The bundled drafters keep their upstream licenses (MTP drafter: MIT, as part of the base checkpoint; DFlash drafter: MIT); see [Provenance of the bundled drafters](#provenance-of-the-bundled-drafters).

## Citation

Please cite the base model:

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```

and, if you use the DFlash drafter:

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
