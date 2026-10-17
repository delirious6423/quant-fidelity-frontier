---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- exl3
- quantized
- qwen
- qflashrt
---

# Qwen3.8-Flash-Next EXL3 K4.25 v1

Calibrated mixed K4/K5 EXL3 routed experts, averaging **4.25 bits per expert
weight**, including all 48 target layers and the native MTP layer. Per layer,
K5 upgrades cover 72 gate, 120 up and 192 down projections; remaining expert
projections use K4. This average describes routed expert storage, not the
whole checkpoint.

Original non-expert tensors are retained, including **BF16 PLE n-gram tables**.
The 22 safetensors shards contain 302,616 tensors and 179,366,664,184 bytes of
tensor payload (179,406,710,432 bytes including shard headers). The original
license, tokenizer, chat template and processor assets are included.

## Compatibility

Produced with the [GPTQModel fork](https://github.com/tpurtell/GPTQModel) at
`339cccf0c4b62440b572440a7d06608ed9e4a165`. Native execution is being developed
in [QFlashRT](https://github.com/tpurtell/qflashrt) for RTX PRO 6000 Blackwell
and DGX Spark. The runtime exports separate native artifacts from this
checkpoint and reads PLE tables from disk. This is not a claim of plug-and-play
support in stock Transformers or every EXL3 loader.

Expert activations remain 16-bit. QFlashRT's balanced/accurate profiles select
FP8/BF16 KV respectively; KV storage is not part of this quantized checkpoint.
Vision weights and original processor assets are retained, but end-to-end
vision and complete serving qualification remain in progress.

## Validation

An independent byte audit checked all 301,056 packed expert tensors against
49 compact runtime expert artifacts and all 1,560 non-expert tensors against
the authenticated original source. Configuration, tokenizer/processor assets
and all 75,264 packed projection descriptions were audited separately.

An exploratory held-out evaluation covers 89 records and 67,477 next-token
predictions, using original framework block math with the GPTQModel EXL3
projection backend. These runs evaluate only the 48 target layers, with MTP
disabled. The EXL3 MTP layer is included in this release but needs separate
acceptance, quality and speed evaluation. Lower perplexity is better on this fixed test:

| Device | Original BF16 reference | EXL3 framework reference |
| --- | ---: | ---: |
| RTX PRO 6000 Blackwell | 3.87319 | 3.87893 |
| DGX Spark / GB10 | 3.89576 | 3.85601 |

These results include backend numerical effects, are hardware-specific, and
are not a general model ranking or task-quality certification. Coding,
reasoning, long-context, tools, vision and MTP quality qualification continues.

## Provenance

- Original model revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Source identity: `1173fcfcb79c115f9a9339219fc6386bdda636f49fa42cb52e86d2b36045679f`.
- Quantization run: `d8413370b3e143a6925d73f5ff441eee9d832506ba1ff66f360b0362219f7bae`.
- Safetensors index SHA256: `4be63d793de784e4e89b1a1aba3809109b3c88c87f58f0b79b88251000c5c148`.
- Calibration and runtime details: [quantization documentation](https://github.com/tpurtell/qflashrt/blob/main/docs/quantization-run.md).

The source **Qwen Community License 1.0** applies; see the included LICENSE.
