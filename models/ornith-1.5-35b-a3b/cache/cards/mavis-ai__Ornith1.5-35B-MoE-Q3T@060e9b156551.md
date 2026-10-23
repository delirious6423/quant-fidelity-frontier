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
- q3t
- apple-silicon
- speculative-decoding
- mtp
- dflash
---

# mavis-ai/Ornith1.5-35B-MoE-Q3T

Q3T build of [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for **R.E.V.I.S.** on Apple silicon, bundled with its speculative-decoding drafters (MTP and DFlash).

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
| Format | Q3T: K3 routed experts + N6 dense |
| Routed experts | 3-bit trellis-coded (K3) |
| Dense / attention / shared-expert / embedding / LM head | 6-bit affine, group size 64 (N6) |
| Routers and selection layers | BF16 |
| Vision tower | BF16 |
| Calibration data | none (data-free) |
| Weight size | 15.15 GB (decimal), 14.1 GiB |
| Bundled drafters | MTP `Ornith1.5-35B-A3B-MTP-Draft-Q8` (+ `ProposalHead`), `Ornith1.5-35B-A3B-DFlash-Q8` (1.57 GB together) |
| KV cache (runtime) | 8-bit |

## Bundle layout

```text
.
├── README.md, .gitattributes, assets/ (upstream images)
├── config.json, configuration.json, generation_config.json, quantization_policy.json
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, chat_template.jinja
├── preprocessor_config.json, processor_config.json, video_preprocessor_config.json
├── model-core-00001-of-00001.safetensors        2.91 GB   dense, attention, embeddings, routers, vision
├── model-experts-0000{1..8}-of-00008.safetensors  8 x 1.53 GB   trellis-coded routed experts
├── model.safetensors.index.json
├── dense_n6_manifest.json, q4t_manifest.json, q4t-split-lineage.json, q4t_sources.json
└── drafts/
    ├── SHA256SUMS
    ├── Ornith1.5-35B-A3B-MTP-Draft-Q8/              MTP drafter, 8-bit affine (g64)
    ├── Ornith1.5-35B-A3B-MTP-Draft-Q8-ProposalHead/ proposal head for the MTP drafter
    └── Ornith1.5-35B-A3B-DFlash-Q8/   DFlash block drafter, 8-bit affine (g64)
```

The core file and the expert files are split so the runtime can map experts independently; expert tensors are byte-identical to the unsplit source pack (`q4t-split-lineage.json`). Checksums for the files under `drafts/` are in `drafts/SHA256SUMS`; per-file SHA-256 values for the root files are recorded in `q4t_manifest.json` (the converter's record: its `files` table also lists local download-cache entries (`.cache/…`), which are not part of the repository, and its flags describe the conversion step, which verified the effective weights, not an engine forward pass). `quantization_policy.json` is carried over from the affine baseline build and identifies the source repository; the quantization actually applied is the one described below and in `q4t_manifest.json`.

## Quantization (Q3T)

The T family (Q5T, Q4T, Q3T) is a fixed, data-free recipe: one width per module class, chosen by a written rule rather than by looking at evaluation data. The formats differ only in the expert rate K and the dense width N. This one is **K3 + N6**.

- **Routed experts, K3.** Each expert matrix is stored as 16x16-tile trellis codes (3 bits per weight, MCG codebook), with a 128-point Hadamard rotation and per-channel sign vectors (`suh`, `svh`). Encoding minimizes rotated-weight MSE; there is no calibration set, Hessian rounding, importance matrix or quantization-aware training. The trellis decoder and its kernels are part of the R.E.V.I.S. inference engine.
- **Everything else linear, N6.** Attention and linear-attention projections, shared expert, token embeddings and LM head are 6-bit affine, group size 64 (N6), quantized from the BF16 source with `mx.quantize` (round to nearest).
- **BF16 kept as-is.** Router (`mlp.gate`), `shared_expert_gate`, linear-attention `in_proj_a`/`in_proj_b`, norms, and the whole vision tower. Expert selection is a discontinuous top-k decision, so these tensors are never quantized.

| Format | Routed experts | Dense |
|---|---|---|
| Q5T | K5 | N8 (8-bit affine, g64) |
| Q4T | K4 | N6 (6-bit affine, g64) |
| Q3T | K3 | N6 (6-bit affine, g64) |

## Benchmarks

### Interim figures (to be replaced)

Simple check run by mavis-ai on the R.E.V.I.S. inference engine, kept only until a full benchmark replaces it: KL divergence of next-token distributions from the 8-bit (Q8) build of the same model, teacher-forced on 2 raw-text documents (1,024 scored positions; mean over positions, lower is better), and top-1 agreement with Q8. This is a small distribution-distance check, not a task benchmark; on this small set the Q3T and Q4 means are close (Q3T has the lower median, 0.0064 vs 0.0131, and the heavier tail). Q5T could not be scored on the engine yet (no K5 kernel), so its row is empty.

| Arm | KLD vs Q8 | Top-1 agreement with Q8 | Weight size |
|---|---:|---:|---:|
| Q5T | not measured | not measured | 23.8 GB |
| Q4T | 0.0447 | 0.929 | 19.2 GB |
| Q4 (affine, context) | 0.0856 | 0.897 | 20.7 GB |
| Q3T (this repository) | 0.0904 | 0.905 | 15.15 GB |

## Runtime

- Runtime: the inference engine built into **R.E.V.I.S.**, a native Swift / C++20 / MLX engine with ahead-of-time compiled Metal kernels for Apple silicon. It has no Python, localhost server or just-in-time Metal in the inference path.
- Speculative decoding: **MTP is the default drafter**; **DFlash is optional**. Both are bundled under `drafts/`. Each drafter is only valid with this target. The `ProposalHead` is pinned to this pack's header digest; on any other target the runtime ignores it and keeps the exact output head. The MTP drafter is the replacement head from shisa.ai rather than the MTP module shipped inside the base checkpoint: on our runtime it was accepted clearly more often than the shipped head (see the release notes of this build).
- Losslessness: with greedy decoding, output with a drafter is intended to equal plain decoding of this target, up to the rounding ties that the runtime certifies. Drafters are an execution-speed feature of the runtime and do not change the target's output.
- Chips: the runtime targets M1 to M5; development and verification so far were on M4- and M5-family Macs.
- Sampling defaults in `generation_config.json` follow the base model (`temperature 1.0`, `top_k 20`, `top_p 0.95`).
- Speed depends on the runtime and the machine, not on this model file, and is therefore not reported here.

## Memory

Weights on disk and in memory: 15.15 GB. The bundled drafters add 1.57 GB if all are loaded. The KV cache is 8-bit and grows with context; the runtime admits a load against the available memory budget and reports the largest context that fits. Budget for weights plus drafters plus KV and transient buffers when choosing a Mac.

## Provenance of the bundled drafters

Everything in this repository ships together, so the origin of every bundled component is listed here. All bundled drafters are derivatives of their upstream releases; mavis-ai claims no ownership of any of them, and each keeps its upstream license.

| Directory | What it is | Original authors and upstream repository | Upstream revision |
|---|---|---|---|
| (root) | the target model, converted to this format | Ornith team, [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | `10fbf86fed7ecee4a061f8b499a618f46001cac1` |
| `drafts/Ornith1.5-35B-A3B-MTP-Draft-Q8/` | MTP drafter: the replacement MTP head released by shisa.ai (initialised from the Apache-2.0 MTP head of [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), re-aligned to Ornith-1.5-35B-A3B by full-vocabulary KL distillation), converted to the MLX drafter layout and quantized to 8-bit affine (group size 64); `fc`, router, shared-expert gate and norms stay BF16. License: Apache License 2.0 (`LICENSE`, `NOTICE` inside the directory) | shisa.ai, [`shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY) | `2b19b31bfe1659c6b0d9459ec3cbd87e34a322ef` |
| `drafts/Ornith1.5-35B-A3B-MTP-Draft-Q8-ProposalHead/` | proposal head for the MTP drafter, made by mavis-ai (see below) | derived from the BF16 `lm_head.weight` of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | `10fbf86fed7ecee4a061f8b499a618f46001cac1` |
| `drafts/Ornith1.5-35B-A3B-DFlash-Q8/` | DFlash block-diffusion drafter, 8-bit affine (group size 64); `fc` and norms stay BF16. License: MIT, as declared upstream | Ornith team, [`ornith-ai/Ornith-1.5-35B-A3B-DFlash`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash); DFlash method and code: [z-lab/dflash](https://github.com/z-lab/dflash), paper [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | `942d20fee84d04c9ecbf18e9c6fc4144914a1d5d` |

**ProposalHead (derived artifact).** `proposal_head.safetensors` is a 3-bit affine (group size 64) copy of the base model's BF16 output head (`lm_head.weight`). The runtime uses it only to choose which blocks of 128 token ids the MTP drafter's proposal is searched in; the chosen rows are then rescored through the exact output head of this pack, and the target model verifies every proposal, so it cannot change the output. `proposal_head.json` pins it to the MTP drafter and to this pack (header digests); on any other target the runtime ignores it and keeps the exact head. No training was involved.

The `README.md` inside each drafter directory restates its origin and license. Upstream benchmark material is not reproduced in this repository.

## License and attribution

mavis-ai claims no ownership of this model. This repository redistributes a quantized derivative of [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), released by the Ornith team; its model card declares the MIT License (the upstream repository ships no separate license file at the source revision).

**Modification notice:** the official checkpoint was converted to the Q3T layout described above (K3 trellis-coded routed experts, 6-bit affine dense tensors, BF16 routers and vision tower). No fine-tuning, additional training or architecture change was applied.

The bundled drafters keep their upstream licenses (MTP drafter: Apache License 2.0; DFlash drafter: MIT); see [Provenance of the bundled drafters](#provenance-of-the-bundled-drafters).

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
