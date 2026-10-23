---
library_name: ignis
pipeline_tag: text-generation
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ignis
  - qwen3.8
  - trellis
  - exl3
  - dflash2
  - speculative-decoding
  - cuda
  - blackwell
  - 16gb
  - conversational
---

# Qwen3.8-27B, 3-bit trellis body, for ignis on 16 GB cards

This repository holds [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) packed for
**[ignis](https://github.com/gpillon/ignis)** for the 16 GB Blackwell cards (RTX 5060 Ti 16 GB,
5070 Ti, 5080). It is the second artifact of the 27B in ignis, beside the 32 GB one
([gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer)).

- **The body and the output head** are turboderp's EXL3 quant
  [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3), branch
  `SC_3.00bpw_H4` (revision `86b95530f9d365d90dd24d01a997a8de0cdf6c55`), repacked bit for bit: every
  matrix's trellis, `suh` and `svh` tensors are stored as that branch stores them. Nothing was
  re-quantized.
- **Everything else** is the 32 GB artifact's object, byte for byte: the token embedding (8-bit), the
  norms and Gated DeltaNet parameters, the DFlash2 drafter and its draft head, the MTP layer, the vision
  tower, the tokenizer and chat template.

The file is a `.ninfer` v2 container that only ignis reads. It is not a Transformers checkpoint,
safetensors, GGUF or EXL3 model.

## How to run

ignis serves this file from the release that wires the 16 GB card class
([gpillon/ignis#339](https://github.com/gpillon/ignis/issues/339)). There is no command to run it yet.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `qwen3_8_27b_trellis_sc30-v2.ninfer` | 13,379,877,124 (12.46 GiB) | The model |
| `qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json` | 309,495 | The conversion record: sources by sha256, every matrix's K and reconstruction error, every copied object's sha256 |
| `qwen3_8_27b_trellis_sc30-v2.ninfer.sha256` | | The model's checksum, in `sha256sum` format |
| `LICENSE` | | Apache License 2.0, as distributed with Qwen/Qwen3.8-27B |

sha256 of the model:

```
2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c
```

## What is inside

| Part | Format | Tensors | Bytes |
|---|---|---:|---:|
| Body: 48 Gated DeltaNet and 16 attention layers, their linears and MLPs | Trellis code, `mul1` codebook, a bit width K per matrix: K=2: 34, K=3: 297, K=4: 66, K=5: 4; 3.00 bits per weight on average | 400 | 9,135,915,008 |
| Output head | Trellis, K=4 | 1 | 636,207,104 |
| Token embedding (248,320 × 5,120) | 8-bit, groups of 32, fp16 scales | 1 | 1,350,860,800 |
| DFlash2 drafter (5 layers) and its 131,072-row draft head | NVFP4 weights, BF16 norms; the head Q4 | 68 | 1,439,902,344 |
| MTP layer, vision tower, norms, Gated DeltaNet parameters | as in the 32 GB artifact | 746 | 799,465,376 |
| Tokenizer, chat template, configs | the upstream files, embedded | 6 | 12,837,177 |

The fused projections of the 32 GB artifact are split here as the quant stores them (`in_proj_qkv` and
`in_proj_z`, `q_proj` / `k_proj` / `v_proj`, `gate_proj` and `up_proj`), each with its own scales.

**Device memory**, from ignis's binder on this file, with the token embedding in host memory: the
language model 9.83 GB; with the DFlash2 drafter and its head 11.26 GB; with the vision tower too
11.56 GB. The KV cache and the kernels' workspaces come on top.

## Measurements

The quant's weights against the BF16 checkpoint, measured layer by layer in a PyTorch harness with BF16
activations (not in ignis), on ignis's KLD windows (16 × 4,096 tokens of code, prose and chat):

| | KLD mean | Top-1 agreement | Perplexity |
|---|---:|---:|---:|
| This body (SC_3.00bpw_H4 weights) | 0.0340 | 93.88% | 2.733 |
| The 32 GB artifact's NVFP4 weights | 0.0458 | 92.92% | 2.744 |
| BF16 | | | 2.690 |

- On a held-out set, perplexity rose 8.6% against BF16 on Python source and 1.4-3.7% in eight other
  domains (prose, chat, code, English, Italian, German, Japanese, Chinese).
- MMLU-Pro proxy (281 questions): 64.77%, against 68.33% for BF16.
- DFlash2 tokens per round on this quant, in ExLlamaV3's own DFlash2 path, against an EXL3 5.0-bit
  control: 1.00-1.06 of the control's, on agent turns at 8K, 64K and 128K tokens and on coding prompts.
- Reconstruction error of each matrix against its BF16 original (relative Frobenius norm): 0.150 on
  average, 0.338 at most, 0.0765 for the output head.

Details:
[ignis finding 2026-10-21](https://github.com/gpillon/ignis/blob/main/docs/findings/2026-10-21-exl3-3bpw-27b-on-our-text-and-dflash2.md).

## Provenance

- **Body and head:** turboderp/Qwen3.8-27B-exl3, branch `SC_3.00bpw_H4`, revision
  `86b95530f9d365d90dd24d01a997a8de0cdf6c55`, quantized with ExLlamaV3 by turboderp. Both shards are
  pinned by sha256 in the conversion record.
- **Everything else:** gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer, file
  `qwen3_8_27b_nvfp4full-v2.ninfer` (sha256 `abb1e120…fd9af`), pinned with its graft record.
- **Reconstruction errors** measured against Qwen/Qwen3.8-27B at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` with ExLlamaV3 1.5.3's reconstruction.
- **Converter:** [`tools/27b-16gb-converter`](https://github.com/gpillon/ignis/tree/main/tools/27b-16gb-converter)
  at ignis commit `cf5e328`; the design is
  [spec 27b-16gb/01](https://github.com/gpillon/ignis/blob/main/docs/specs/27b-16gb/01-the-16gb-artifact.md).
  Packing the same sources twice gives the same file.

## License

Apache License 2.0, as for Qwen/Qwen3.8-27B and for turboderp's quant. The trellis records are
turboderp's quantization of Qwen's weights; ExLlamaV3 is MIT. ignis itself is Apache-2.0.
