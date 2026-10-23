---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- gemma4
- moe
- quantized
- trellis
- q4t
- apple-silicon
- speculative-decoding
- mtp
- dflash
---

# mavis-ai/Gemma4-26B-MoE-Q4T

Q4T build of [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) for **R.E.V.I.S.** on Apple silicon, bundled with its speculative-decoding drafters (MTP and DFlash).

mavis-ai claims no ownership of this model. It is a data-free quantized derivative of the official Gemma 4 26B A4B checkpoint; all rights in the underlying model remain with its original authors, and it is redistributed under the Apache License 2.0. It is not a fine-tune: no training of any kind was applied.

> **Compatibility.** The routed experts are stored as trellis-coded tensors (`.trellis`, `.suh`, `.svh`) and need a special trellis decoding computation. This model therefore currently does **not** run on `mlx-lm` (nor on `mlx-vlm`, `transformers` or vLLM). It runs in **R.E.V.I.S. v1.3.0 and later**, whose built-in inference engine provides the kernels.

## R.E.V.I.S.

This model is made for **R.E.V.I.S.**, a 100% local Cognitive OS for Multi-Agentic AI on Mac. This model is supported from **R.E.V.I.S. v1.3.0**.

- **🌐 Official Website:** [https://mavis-ai.co.jp/revis/](https://mavis-ai.co.jp/revis/)
- **🐦 Follow our updates on X:** [@mavis_ai_jp](https://x.com/mavis_ai_jp)

## Summary

| | |
|---|---|
| Base model | Gemma 4 26B A4B (multimodal MoE; 30 layers, hidden size 2816, 128 routed experts with 8 active, context length 262144) |
| Format | Q4T: K4 routed experts + N6 dense |
| Routed experts | 4-bit trellis-coded (K4) |
| Dense MLP / attention / token embeddings (tied LM head) | 6-bit affine, group size 64 (N6) |
| Routers | BF16 |
| Vision tower and multimodal projection | BF16 |
| Calibration data | none (data-free) |
| Weight size | 15.65 GB (decimal), 14.6 GiB |
| Bundled drafters | MTP `Draft-Q8` (+ `ProposalHead`), `DFlash-Q8` (1.11 GB together) |
| KV cache (runtime) | 8-bit |

## Bundle layout

```text
.
├── README.md, LICENSE, .gitattributes
├── config.json, generation_config.json
├── tokenizer.json, tokenizer_config.json, chat_template.jinja, processor_config.json
├── model-core-00001-of-00001.safetensors        3.11 GB   dense, attention, embeddings, routers, vision
├── model-experts-0000{1..6}-of-00006.safetensors  6 x 2.09 GB   trellis-coded routed experts
├── model.safetensors.index.json
├── dense_n6_manifest.json, q4t_manifest.json, q4t-split-lineage.json, q4t_sources.json
└── drafts/
    ├── SHA256SUMS
    ├── Gemma4-26B-Draft-Q8/                 MTP drafter, 8-bit affine (g64), 0.45 GB weights
    ├── Gemma4-26B-Draft-Q8-ProposalHead/    proposal head for the MTP drafter, 0.12 GB
    └── Gemma4-26B-MoE-DFlash-Q8/            DFlash block drafter, 8-bit affine (g64), 0.50 GB
```

The core file and the expert files are split so the runtime can map experts independently; expert tensors are byte-identical to the unsplit source pack (`q4t-split-lineage.json`). Checksums for the files under `drafts/` are in `drafts/SHA256SUMS`; per-file SHA-256 values for the root files are recorded in `q4t_manifest.json`.

## Quantization (Q4T)

The T family (Q5T, Q4T, Q3T) is a fixed, data-free recipe: one width per module class, chosen by a written rule rather than by looking at evaluation data. The formats differ only in the expert rate K and the dense width N. This one is **K4 + N6**.

- **Routed experts, K4.** Each expert matrix is stored as 16x16-tile trellis codes (4 bits per weight, MCG codebook), with a 128-point Hadamard rotation and per-channel sign vectors (`suh`, `svh`). Encoding minimizes rotated-weight MSE; there is no calibration set, Hessian rounding, importance matrix or quantization-aware training. The trellis decoder and its kernels are part of the R.E.V.I.S. inference engine. Expert width 704 is not a multiple of 128 and is stored padded to 768.
- **Everything else linear, N6.** Attention projections, the dense MLP and token embeddings (the LM head is tied to them) are 6-bit affine, group size 64, quantized from the BF16 source with `mx.quantize` (round to nearest).
- **BF16 kept as-is.** Router (`router.proj`, `router.scale`, `router.per_expert_scale`), norms, layer scalars, and the vision tower with its projection. Expert selection is a discontinuous top-k decision, so these tensors are never quantized.

| Format | Routed experts | Dense |
|---|---|---|
| Q5T | K5 | N8 (8-bit affine, g64) |
| Q4T | K4 | N6 (6-bit affine, g64) |
| Q3T | K3 | N6 (6-bit affine, g64) |

## Benchmarks

The question these benchmarks answer is how well each quantization suppresses degradation relative to the full-precision model (Full, BF16). All arms are scored on the same inputs against the same BF16 reference. The final numbers will come from a single benchmark run and are not available yet; the table below is intentionally empty.

| Metric | Full (BF16) | Q8 | Q6 | Q5 | Q4 | Q5T | Q4T | Q3T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KLD vs Full (lower is better) |   |   |   |   |   |   |   |   |
| Top-1 agreement with Full |   |   |   |   |   |   |   |   |
| Accuracy benchmarks (to be defined) |   |   |   |   |   |   |   |   |

### Interim figures (to be replaced)

Earlier internal measurement, kept only until the benchmark above is filled in. KL divergence of next-token distributions from the BF16 Gemma 4 26B A4B reference, teacher-forced, mean over scored positions (lower is better). Evaluation set: 48 documents, 18,040 scored positions. These are distribution-distance measurements on a small set, not task benchmarks, and they are **interim**: the benchmark table above supersedes them.

| Arm | KLD vs BF16 | Weight size |
|---|---:|---:|
| Q8 | 0.0110 | n/a |
| Q5T | 0.0136 | 19.36 GB |
| Q6 | 0.0268 | 21.85 GB |
| Q4T (this repository) | 0.0373 | 15.64 GB |
| Q5 | 0.0638 | 18.60 GB |
| Q3T | 0.0650 | 12.53 GB |
| Q4 | 0.2367 | 15.54 GB |

## Runtime

- Runtime: the inference engine built into **R.E.V.I.S.**, a native Swift / C++20 / MLX engine with ahead-of-time compiled Metal kernels for Apple silicon. It has no Python, localhost server or just-in-time Metal in the inference path.
- Speculative decoding: **MTP is the default drafter**; **DFlash is optional**. Both are bundled under `drafts/`. Each drafter is only valid with this target. The `ProposalHead` is pinned to the bundled MTP drafter, whose own tied embedding is the exact head.
- Losslessness: with greedy decoding, output with a drafter is intended to equal plain decoding of this target, up to the rounding ties that the runtime certifies. Drafters are an execution-speed feature of the runtime and do not change the target's output.
- Chips: the runtime targets M1 to M5; development and verification so far were on M4- and M5-family Macs.
- Sampling defaults in `generation_config.json` follow the base model card (`temperature 1.0`, `top_k 64`, `top_p 0.95`).
- Speed depends on the runtime and the machine, not on this model file, and is therefore not reported here.

## Memory

Weights on disk and in memory: 15.65 GB. The three drafters add 1.11 GB if all are loaded. The KV cache is 8-bit and grows with context; the runtime admits a load against the available memory budget and reports the largest context that fits. Budget for weights plus drafters plus KV and transient buffers when choosing a Mac.

## Provenance of the bundled drafters

Everything in this repository ships together, so the origin of every bundled component is listed here. All bundled drafters are derivatives of their upstream releases; mavis-ai claims no ownership of any of them, and each keeps its upstream license (Apache License 2.0).

| Directory | What it is | Original authors and upstream repository | Upstream revision |
|---|---|---|---|
| (root) | the target model, converted to this format | Google DeepMind, [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | `462a98a12e28e2cbcfccaf78fe41e3e50235e6ae` |
| `drafts/Gemma4-26B-Draft-Q8/` | MTP drafter (the Gemma 4 26B A4B assistant model), quantized to 8-bit affine (group size 64); tied embedding / output head 8-bit, `pre_projection`, `post_projection` and norms BF16 | Google DeepMind, [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) | `b7696b57a62c7a24a177254d1135fd6c0e650792` |
| `drafts/Gemma4-26B-Draft-Q8-ProposalHead/` | proposal head for the MTP drafter, made by mavis-ai (see below) | derived from the BF16 tied token embedding (`model.embed_tokens.weight`) of [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) | `b7696b57a62c7a24a177254d1135fd6c0e650792` |
| `drafts/Gemma4-26B-MoE-DFlash-Q8/` | DFlash block-diffusion drafter, 8-bit affine (group size 64); `fc` and norms stay BF16 | Z-Lab, [`z-lab/gemma-4-26B-A4B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash); DFlash method and code: [z-lab/dflash](https://github.com/z-lab/dflash), paper [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | `77d4202772dfe50b2396ec7bac9cfffc7b9e7057` |

**ProposalHead (derived artifact).** `proposal_head.safetensors` is a 3-bit affine (group size 64) copy of the assistant drafter's BF16 tied token embedding, which is also the drafter's output head. The runtime uses it only to choose which blocks of 128 token ids the MTP drafter's proposal is searched in; the chosen rows are then rescored through the drafter's own 8-bit output head, and the target model verifies every proposal, so it cannot change the output. Because the exact head lives in the drafter itself, `proposal_head.json` pins the drafter (header digest) and lists no target pack. No training was involved.

The `README.md` inside each drafter directory restates its origin and license. Upstream benchmark material is not reproduced in this repository.

## License and attribution

mavis-ai claims no ownership of this model. This repository redistributes a quantized derivative of [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it), released by Google under the Apache License 2.0. A copy of the license is in `LICENSE`.

**Modification notice:** the official checkpoint was converted to the Q4T layout described above (K4 trellis-coded routed experts, 6-bit affine dense tensors, BF16 routers and vision tower). No fine-tuning, additional training or architecture change was applied.

The bundled drafters keep their upstream licenses (Apache License 2.0); see [Provenance of the bundled drafters](#provenance-of-the-bundled-drafters).

## Citation

Please cite the base model:

```bibtex
@misc{gemmateam2026gemma4,
      title={Gemma 4 Technical Report},
      author={Gemma Team},
      year={2026},
      eprint={2607.02770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.02770},
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

## Tokenizer note (2026-10-23)

`tokenizer.json` here is Google's published file plus the 70 HTML-tag pieces (ids 168-237, e.g. `</div>` = 235) of Google's reference SentencePiece tokenizer (`tokenizer_gemma4.model`), added as `special: false` added tokens. The published file never produces these pieces, so `</div>` was encoded as `</` `div` `>` although the model was trained with the whole piece. Nothing else changed. Original sha256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` (the value `q4t_manifest.json` records, where present); this file sha256 `e81d757e09e630458df2c759d951d11333716f7461100820969d17821033b263`.
