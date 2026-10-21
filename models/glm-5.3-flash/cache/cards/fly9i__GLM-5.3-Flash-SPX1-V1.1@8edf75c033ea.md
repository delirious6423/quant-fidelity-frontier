---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: spark-engine
pipeline_tag: image-text-to-text
license: other
license_name: shapleymcg-license-1.0
license_link: https://github.com/brandonmmusic-max/shapleymcg/blob/main/LICENSE
tags:
  - shapleymcg
  - glm
  - exl3
  - trellis
  - mixed-precision
  - spx1
  - multimodal
---

# GLM-5.3-Flash SPX1 V1.1

GLM-5.3-Flash quantized for two NVIDIA DGX Spark (GB10) nodes running tensor-parallel 2. The routed experts average about 4.0 bits per weight, with a mixed per-expert bit width. The other linear layers use 6-bit SPX1 codes. Total size is 156 GB.

> **Runtime:** this checkpoint runs only on our own inference engine (spark-engine, `spx1` branch). The engine is not public yet.
> The SPX1 non-expert format and the mixed-K expert layout cannot be loaded by vLLM, ExLlamaV3, or Transformers.

## Contents

| Part | Format | Notes |
| --- | --- | --- |
| Routed experts (`model-e000NN-experts.safetensors`, 43 shards) | EXL3 trellis codes, MCG codebook | Per-expert K ∈ {3, 4, 5, 6}: 22.6 / 51.3 / 25.7 / 0.4 % of experts. Average is 4.04 bpw stored and 4.00 bpw actually read per 8-token decode window. |
| Non-expert linears (`model-s00001-spx-nonexpert.safetensors`) | SPX1 "SG6": scalar INT6 + Hadamard rotation | Attention (KDA / MLA), dense and shared-expert MLPs, and the LM head. Codes are stored as 16×128 tiles, with one FP16 scale per 64-weight group. |
| Everything else (`model-00001-of-00001.safetensors`) | Source dtype, unchanged | Embeddings, norms, MHC mixing parameters, router, and vision tower. Copied byte-for-byte from the upstream tr3 checkpoint. |

The engine resolves tensors as follows. Files are read in sorted order and later files override earlier ones. An SPX tensor `N.spx_q` / `N.spx_s` / `N.spx_sign` replaces its higher-precision original `N`.

### How it was made

- **Experts:** fully re-quantized from the official FP8 checkpoint (`zai-org/GLM-5.3-Flash`). The steps were:
  1. Collect per-layer Hessians from the experts' inputs while running the FP8 model as the teacher.
  2. Shrink each expert's Hessian toward the layer average (n0 ≈ 2048).
  3. Share one input-side sign/scale vector (`suh`) across all gate/up projections of a layer. The `suh` vector is only the input-side sign/scale; the trellis codes stay separate per expert.
  4. Assign each expert's K by activation energy × routing frequency, keeping total read bandwidth equal to uniform 4-bit.
  5. Encode the codes with the ExLlamaV3 encoder (v0.0.43, MCG codebook), using our own LDLQ/Viterbi path that is bit-identical to ExLlamaV3's.
- **Non-expert linears:** GPTQ (damp 0.05) with Hadamard rotation, using non-expert Hessians collected over 138K calibration tokens.
- **Residue tensors:** taken unchanged from [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw).

## Quality

Both reference points are the official FP8 model. KL is measured on a held-out evaluation set that was not used for calibration. The numeric floor of the measurement is about 0.0045.

| Checkpoint | eval KL vs FP8 | top-1 agreement | GSM8K (200) |
| --- | --- | --- | --- |
| **SPX1 V1.1 (this)** | **0.0171** | **96.5 %** | **98.5 %** |
| tr3 4bpw (upstream, uniform K4) | 0.0231 | 95.6 % | – |

## Speed

Measured on 2× DGX Spark, TP2 over RoCE, single stream, thinking off, greedy decoding. Speculative decoding uses a 4-bit GPTQ DFlash2 drafter, which is **not included** (see below).

| | |
| --- | --- |
| Decode, 30 mixed prompts | 49.8 tok/s (2.76 tokens accepted per round) |
| Prefill, 8K / 32K tokens | 1288 / 1282 tok/s (INT8 non-expert prefill path) |
| Expert pool load | about 15 s per rank |

## Not included

- **Drafter.** The DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2`) is licensed CC-BY-NC-ND-4.0. Our GPTQ 4-bit version of it is a derivative work and cannot be redistributed. Get the original from its own repository.
- **Ablation.** These weights are **not ablated**. Any refusal ablation in our deployment is applied at run time by the engine and is not stored in this checkpoint.

## Files kept from the upstream release

- `LICENSE` (ShapleyMcg License v1.0) and `THIRD_PARTY_NOTICES.md` are kept unchanged, as the license requires.
- `PROVENANCE-upstream-tr3.md` is the upstream `PROVENANCE.md`, unchanged in content.
- `quantization_config.json` and `exl3-mcg-storage-abi.json` are also unchanged in content. They describe the upstream **uniform-K4** expert encoding and are kept for provenance. The per-expert K of V1.1 is given by each expert's trellis tensor shape (K = last dimension / 16).
- The upstream vLLM runtime files are omitted. They do not apply to this checkpoint: `compose.yaml`, `serve-glm53-sm120-tp2.sh`, and `runtime-results-v44.json`.
- `SHA256SUMS` lists the SHA-256 of every file in this repository.

## Attribution

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```

Also built on:
- GLM-5.3-Flash by Z.ai (MIT License).
- The EXL3 trellis encoder by turboderp and the ExLlamaV3 contributors.
- QTIP / QuIP# by Tseng et al.

Our own changes are listed under "How it was made": expert re-quantization and bit allocation, the SPX1 non-expert format, and the engine. Everything else comes from the upstream works listed above.
