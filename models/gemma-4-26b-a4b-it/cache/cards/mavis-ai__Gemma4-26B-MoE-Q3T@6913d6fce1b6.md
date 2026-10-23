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
- q3t
- apple-silicon
- speculative-decoding
- mtp
- dflash
---

# mavis-ai/Gemma4-26B-MoE-Q3T

Q3T build of [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) for **R.E.V.I.S.** on Apple silicon, bundled with its speculative-decoding drafters (MTP and DFlash).

mavis-ai claims no ownership of this model. It is a data-free quantized derivative of the official Gemma 4 26B A4B checkpoint; all rights in the underlying model remain with its original authors, and it is redistributed under the Apache License 2.0. It is not a fine-tune: no training of any kind was applied.

> **Compatibility.** The routed experts are stored as trellis-coded tensors (`.trellis`, `.suh`, `.svh`) and need a special trellis decoding computation. This model therefore currently does **not** run on `mlx-lm` (nor on `mlx-vlm`, `transformers` or vLLM). It is made for **R.E.V.I.S.**, whose built-in inference engine provides the kernels; support for this Q3T build comes in a R.E.V.I.S. release after v1.3.0.

## R.E.V.I.S.

This model is made for **R.E.V.I.S.**, a 100% local Cognitive OS for Multi-Agentic AI on Mac. Support for this Q3T build comes in a R.E.V.I.S. release after v1.3.0.

- **🌐 Official Website:** [https://mavis-ai.co.jp/revis/](https://mavis-ai.co.jp/revis/)
- **🐦 Follow our updates on X:** [@mavis_ai_jp](https://x.com/mavis_ai_jp)

## Summary

| | |
|---|---|
| Base model | Gemma 4 26B A4B (multimodal MoE; 30 layers, hidden size 2816, 128 routed experts with 8 active, context length 262144) |
| Format | Q3T: K3 routed experts + N6 dense |
| Routed experts | 3-bit trellis-coded (K3) |
| Dense MLP / attention / token embeddings (tied LM head) | 6-bit affine, group size 64 (N6) |
| Routers | BF16 |
| Vision tower and multimodal projection | BF16 |
| Calibration data | none (data-free) |
| Weight size | 12.53 GB (decimal), 11.7 GiB |
| Bundled drafters | MTP `Draft-Q8` (+ `ProposalHead`), `DFlash-Q8` (1.11 GB together) |
| KV cache (runtime) | 8-bit |

## Bundle layout

```text
.
├── README.md, LICENSE, .gitattributes
├── config.json, generation_config.json
├── tokenizer.json, tokenizer_config.json, chat_template.jinja, processor_config.json
├── model-core-00001-of-00001.safetensors        3.11 GB   dense, attention, embeddings, routers, vision
├── model-experts-0000{1..6}-of-00006.safetensors  6 x 1.57 GB   trellis-coded routed experts
├── model.safetensors.index.json
├── dense_n6_manifest.json, q4t_manifest.json, q4t-split-lineage.json, q4t_sources.json
└── drafts/
    ├── SHA256SUMS
    ├── Gemma4-26B-Draft-Q8/                 MTP drafter, 8-bit affine (g64), 0.45 GB weights
    ├── Gemma4-26B-Draft-Q8-ProposalHead/    proposal head for the MTP drafter, 0.12 GB
    └── Gemma4-26B-MoE-DFlash-Q8/            DFlash block drafter, 8-bit affine (g64), 0.50 GB
```

The core file and the expert files are split so the runtime can map experts independently; expert tensors are byte-identical to the unsplit source pack (`q4t-split-lineage.json`). Checksums for the files under `drafts/` are in `drafts/SHA256SUMS`; per-file SHA-256 values for the root files are recorded in `q4t_manifest.json`. `q4t_manifest.json` is the converter's record: its `files` table also lists a `.gitattributes`, which is not part of this repository (the `.gitattributes` here is the Hub's own), and its flags (`runtime_forward_verified: false`, the `producer` note) describe the conversion step, which verified the effective weights, not an engine forward pass.

## Quantization (Q3T)

The T family (Q5T, Q4T, Q3T) is a fixed, data-free recipe: one width per module class, chosen by a written rule rather than by looking at evaluation data. The formats differ only in the expert rate K and the dense width N. This one is **K3 + N6**.

- **Routed experts, K3.** Each expert matrix is stored as 16x16-tile trellis codes (3 bits per weight, MCG codebook), with a 128-point Hadamard rotation and per-channel sign vectors (`suh`, `svh`). Encoding minimizes rotated-weight MSE; there is no calibration set, Hessian rounding, importance matrix or quantization-aware training. The trellis decoder and its kernels are part of the R.E.V.I.S. inference engine. Expert width 704 is not a multiple of 128 and is stored padded to 768.
- **Everything else linear, N6.** Attention projections, the dense MLP and token embeddings (the LM head is tied to them) are 6-bit affine, group size 64, quantized from the BF16 source with `mx.quantize` (round to nearest).
- **BF16 kept as-is.** Router (`router.proj`, `router.scale`, `router.per_expert_scale`), norms, layer scalars, and the vision tower with its projection. Expert selection is a discontinuous top-k decision, so these tensors are never quantized.

| Format | Routed experts | Dense |
|---|---|---|
| Q5T | K5 | N8 (8-bit affine, g64) |
| Q4T | K4 | N6 (6-bit affine, g64) |
| Q3T | K3 | N6 (6-bit affine, g64) |

## Benchmarks

Every number below compares an arm with the full-precision model (BF16) on the same inputs. The arms were run on GPU research harnesses that decode this family's stored weight codes exactly, not through the R.E.V.I.S. engine. Speed is not reported.

Comparisons with ordinary affine quantization will be added after a re-measurement with standard builds.

### Next-token distribution distance (KLD vs BF16)

Teacher-forced next-token distributions of Gemma 4 26B A4B on 48 documents (18,040 scored positions in three strata: WikiText-2, technical documents, and the model's own responses), full vocabulary, KV cache emulated at 8 bits. KLD is the mean KL divergence from the BF16 distribution (lower is better); top-1 is the share of positions where the arm's most likely token equals BF16's. Brackets are 99.8 % bootstrap confidence intervals over documents.

| Arm | KLD vs BF16 [99.8 % CI] | Top-1 agreement | Weight size |
|---|---:|---:|---:|
| Q5T | 0.0136 [0.0118, 0.0153] | 0.960 | 19.36 GB |
| Q4T | 0.0373 [0.0317, 0.0428] | 0.936 | 15.64 GB |
| **Q3T (this repository)** | 0.0650 [0.0562, 0.0750] | 0.916 | 12.53 GB |

### Task accuracy

Non-thinking chat, greedy decoding (temperature 0), the model's own chat template, no system prompt, no output-length cap (a loop detector ends runaway repetition identically for every arm). GSM8K: all 1,319 test items. MMLU-Pro: 700 items, 50 from each of the 14 categories, zero-shot chain of thought. IFEval: 541 prompts, prompt-level strict. Cells give accuracy in %; BF16 with its 95 % Wilson interval, every other arm with Δ, the paired difference against BF16 on the same items, and its 95 % interval. **BF16 (repeat)** is the same BF16 weights run a second time: batched GPU inference is not bitwise reproducible, so its Δ is the engine's own noise band.

| Task (items) | BF16 | BF16 (repeat) | Q4T | Q3T (this repository) |
|---|---:|---:|---:|---:|
| GSM8K (1,319) | 96.59 [95.5, 97.4] | 96.82<br>Δ +0.23 [−0.03, +0.48] | 96.66<br>Δ +0.08 [−0.42, +0.57] | 96.66<br>Δ +0.08 [−0.46, +0.61] |
| MMLU-Pro (700) | 81.29 [78.2, 84.0] | 81.43<br>Δ +0.14 [−1.42, +1.70] | 81.43<br>Δ +0.14 [−1.61, +1.89] | 80.57<br>Δ −0.71 [−2.59, +1.16] |
| IFEval (541) | 89.65 [86.8, 91.9] | 89.83<br>Δ +0.18 [−1.39, +1.76] | 89.09<br>Δ −0.55 [−2.37, +1.26] | 88.54<br>Δ −1.11 [−2.88, +0.66] |

Neither Q4T nor Q3T differs from BF16 significantly: every 95 % interval contains 0. On this model the BF16 repeat band is narrow (at most 0.23 points), and three differences lie outside it without reaching significance: Q3T MMLU-Pro −0.71, Q3T IFEval −1.11 and Q4T IFEval −0.55.

### Speculative drafting acceptance (MTP)

Whether the quantized target still accepts the bundled MTP drafter's proposals as often as BF16 does: mean accepted draft tokens per verification round (6 drafted per round), on the same 26 prompts × 3 seeds for every target, with a 95 % bootstrap interval over prompts. Sampled decoding (temperature 1, top-p 1), thinking on; the bundled 8-bit drafter; each target computed from its pack's weights (routed experts decoded to BF16); KV cache BF16; GPU research harness, not the R.E.V.I.S. engine.

| Target | Mean accepted per round | Δ vs BF16 [95 % CI] |
|---|---:|---:|
| BF16 | 3.376 | — |
| Q5T | 3.410 | +0.034 [−0.108, +0.131] |
| Q4T | 3.399 | +0.024 [−0.094, +0.188] |
| Q3T | 3.365 | −0.011 [−0.117, +0.070] |

Every interval contains 0: no loss in draft acceptance is detectable at a resolution of about ±0.1 accepted tokens per round. Weighting every prompt equally gives the same result, and so does the bundled DFlash drafter. BF16 ran one request at a time and the quantized targets four at a time. This is an acceptance statistic, not a speed measurement.

## Runtime

- Runtime: the inference engine built into **R.E.V.I.S.**, a native Swift / C++20 / MLX engine with ahead-of-time compiled Metal kernels for Apple silicon. It has no Python, localhost server or just-in-time Metal in the inference path.
- Verification: the engine's Q3T forward pass was checked against an independent CPU decode of the same trellis weights (whole-model KL at the engine's own operation-order floor), on M4 Max and M5 Max.
- Speculative decoding: **MTP is the default drafter**; **DFlash is optional**. Both are bundled under `drafts/`. Each drafter is only valid with this target. The `ProposalHead` is pinned to the bundled MTP drafter, whose own tied embedding is the exact head.
- Losslessness: with greedy decoding, output with a drafter is intended to equal plain decoding of this target, up to the rounding ties that the runtime certifies. Drafters are an execution-speed feature of the runtime and do not change the target's output.
- Chips: the runtime targets M1 to M5; development and verification so far were on M4- and M5-family Macs.
- Sampling defaults in `generation_config.json` follow the base model card (`temperature 1.0`, `top_k 64`, `top_p 0.95`).
- Speed depends on the runtime and the machine, not on this model file, and is therefore not reported here.

## Memory

Weights on disk and in memory: 12.53 GB. The three drafters add 1.11 GB if all are loaded. The KV cache is 8-bit and grows with context; the runtime admits a load against the available memory budget and reports the largest context that fits. Budget for weights plus drafters plus KV and transient buffers when choosing a Mac.

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

**Modification notice:** the official checkpoint was converted to the Q3T layout described above (K3 trellis-coded routed experts, 6-bit affine dense tensors, BF16 routers and vision tower). No fine-tuning, additional training or architecture change was applied.

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
