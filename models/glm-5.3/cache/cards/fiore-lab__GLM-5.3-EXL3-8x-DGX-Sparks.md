---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model: zai-org/GLM-5.3
tags:
  - exl3
  - glm
  - moe
  - dgx-spark
---

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

# GLM-5.3, EXL3 4 / 5 bits, for eight DGX Sparks

A quantization of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (755B, `glm_moe_dsa`) by Fiore-lab, made from
[zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) at revision 9d2398f, for the recipe that serves it on
eight NVIDIA DGX Spark systems with tensor parallelism 8: https://github.com/fiore-labo/fiore-glm5.3. That recipe's engine is
the only one this checkpoint is laid out for. On eight DGX Sparks it takes a context of 1M tokens (1,048,576).

## What is in it

| Files | What they hold |
|---|---|
| `experts-layer-003` … `077.safetensors` (75) | routed experts of layers 3 to 77: EXL3, 4 bits, codebook mcg, each expert already cut into its eight tensor-parallel shares |
| `nonexperts-5bit.safetensors` | the other linear layers of layers 0 to 77 (attention projections, shared experts, the dense MLPs of layers 0 to 2, the indexer's `wq_b`): EXL3, 5 bits, codebook mul1 |
| `unquantized-bf16.safetensors` | the tensors kept as the BF16 checkpoint stores them: embeddings, `lm_head`, norms, router, `kv_b_proj`, the indexer's `wk` / `weights_proj` / `k_norm`, the final norm (byte for byte the official ones) |
| `mtp-layer-78-*.safetensors` (3) | the MTP layer (layer 78), the drafter the recipe uses by default: routed experts 4 bits mcg, linear layers 5 bits mul1, `eh_proj` and the other unquantized tensors BF16 |
| `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | the official files; `config.json` adds a `quantization_config` that describes the above |
| `model.safetensors.index.json`, `SHA256SUMS` | the tensor index and the checksums of every file but this card |
| `LICENSE` | the GLM-5.3 License |

389.9 GB (363.1 GiB) in all. No draft model other than the MTP layer is included: the optional
incoai/GLM-5.3-DFlash2 drafter is not redistributed here (its card states CC BY-NC-ND 4.0).

## How it was made

With exllamav3 1.6.1:

- Routed experts: 4 bits, codebook mcg, calibrated with exllamav3's default calibration data; then cut into eight
  tensor-parallel shares by whole tiles (no requantization).
- Other linear layers: 5 bits, codebook mul1, calibrated through the model that already holds the 4-bit experts.
- The MTP layer: quantized without calibration data (as exllamav3 converts an MTP layer): experts 4 bits mcg, linear
  layers 5 bits mul1.
- Tensors exllamav3 keeps unquantized are copied from the BF16 checkpoint as stored (BF16, F32 for the router bias).

## Accuracy

KL divergence of the recipe's output distributions with this checkpoint from the original BF16 model's (the official
BF16 weights run by the official implementation, transformers, computed in FP32), with the recipe's serving
configuration (FP8 KV cache):

- Four 2,048-token texts (dialogue, legal, prose, reasoning), the last 1,023 positions each, five runs: mean 0.0133
  (0.0128 to 0.0146); the top-1 token agrees at about 96%.
- Four 8,192-token texts (two English, two code), by position range: 0.012 to 0.019. For scale, the official BF16
  weights computed in BF16 instead of FP32 differ from the same reference by 0.007 to 0.010 over the same ranges.

## Speed

On eight DGX Sparks with the recipe (release r2-102) and its FP8 KV cache, measured with the GPU clock capped at
2.1 GHz (2,100 MHz) on every node, a coding prompt and one request at a time (MTP: median of two runs; DFlash2: one
run). Both drafters work: the MTP layer included here (three drafts a round), or a DFlash2 drafter, which is not
included (its model card states CC BY-NC-ND 4.0, for research and evaluation, with commercial licensing from Inco AI).

| Prompt | Time to first token (MTP / DFlash2) | Decode with the MTP layer | Decode with DFlash2 |
|---|---|---|---|
| 128K (129,436 tokens) | 82.8 s / 85.2 s | 47.5 tokens/s | 53.2 tokens/s |
| 256K (260,230 tokens) | 172.9 s / 177.6 s | 45.7 tokens/s | 52.6 tokens/s |
| 512K (529,885 tokens) | 376.5 s / 386.0 s | 42.8 tokens/s | 47.1 tokens/s |

The recipe's README has the conditions in full: https://github.com/fiore-labo/fiore-glm5.3.

## License

The GLM-5.3 License (`LICENSE`), Copyright (c) 2026 Z.AI, as for the base model: the copyright notice and the permission
notice must be included in all copies or substantial portions. Its clause 2 reads: "If the Licensee or any of its
affiliates operates a Model as a Service business, and the aggregate revenue of the Licensee and its affiliates exceeds
10 billion US dollars (or the equivalent in other currencies) in total over any consecutive 12 months, the Licensee must
pass Z.AI's security review before using the Software or its derivative works for any commercial purpose." Check
whether it applies to you. The recipe's code is Apache-2.0 and licensed separately.
