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

# GLM-5.3, EXL3 3 / 4 / 5 bits, for four DGX Sparks

A quantization of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (755B, `glm_moe_dsa`) by Fiore-lab, made from
[zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) at revision 9d2398f, for the four-node build of the
recipe that serves it on NVIDIA DGX Spark systems (four systems, tensor parallelism 4, decode context parallelism 2):
https://github.com/fiore-labo/fiore-glm5.3. That recipe's engine is the only one this checkpoint is laid out for.

Half of the routed experts of layers 3 to 77 are at 3 bits instead of 4. With this mix and the engine's FP4 KV cache,
the 755B model runs on four DGX Sparks with a context of about 800K tokens (819,200: the engine and the model take up
to 1M, the four Sparks' memory sets the limit). For eight DGX Sparks (a 1M context), the sister repo
[fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks](https://huggingface.co/fiore-lab/GLM-5.3-EXL3-8x-DGX-Sparks) has all routed
experts at 4 bits.

## What is in it

| Files | What they hold |
|---|---|
| `experts-layer-003` … `077.safetensors` (75) | routed experts of layers 3 to 77: EXL3, codebook mcg, the experts listed in `q3-selection.json` at 3 bits and the rest at 4 bits; each expert already cut into its eight tensor-parallel shares (the four-node engine loads two shares per rank) |
| `q3-selection.json` | the 3-bit experts of each layer: the keys are the layer numbers (`"3"` to `"77"`), the values the expert indices |
| `nonexperts-5bit.safetensors` | the other linear layers of layers 0 to 77 (attention projections, shared experts, the dense MLPs of layers 0 to 2, the indexer's `wq_b`): EXL3, 5 bits, codebook mul1 |
| `unquantized-bf16.safetensors` | the tensors kept as the BF16 checkpoint stores them: embeddings, `lm_head`, norms, router, `kv_b_proj`, the indexer's `wk` / `weights_proj` / `k_norm`, the final norm (byte for byte the official ones) |
| `mtp-layer-78-*.safetensors` (3) | the MTP layer (layer 78), usable as the drafter: routed experts 4 bits mcg, linear layers 5 bits mul1, `eh_proj` and the other unquantized tensors BF16 |
| `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | the official files; `config.json` adds a `quantization_config` that describes the above, including the 3 / 4-bit mix |
| `model.safetensors.index.json`, `SHA256SUMS` | the tensor index and the checksums of every file but the three model cards |
| `LICENSE` | the GLM-5.3 License |

The 88 files other than `SHA256SUMS` and the three model cards total 344.6 GB (320.9 GiB). The routed experts take
323.2 GB, against 368.5 GB in the sister repo. The other weight files (`nonexperts-5bit.safetensors`,
`unquantized-bf16.safetensors`, the three MTP files), the tokenizer files, `chat_template.jinja`,
`generation_config.json` and `LICENSE` are byte for byte those of the sister repo. No draft model other than the MTP
layer is included. No DFlash2 drafter is included: it cannot be redistributed.

To check a download, run this in its folder:

```
sha256sum -c SHA256SUMS
```

## How it was made

With exllamav3 1.6.1:

- Which experts are at 3 bits: of the 19,200 routed experts of layers 3 to 77, the least used 50 % (9,600), chosen over all
  layers together from usage statistics across the target domains: agent work, coding, Linux operations and server
  security, each in Japanese and English. 103 to 164 of the 256 experts of a layer are at 3 bits; the shallowest layers
  have the most.
- Routed experts: 3 bits (that 50 %) or 4 bits (the rest, as in the sister repo), codebook mcg, both calibrated with
  exllamav3's default calibration data; then cut into eight tensor-parallel shares by whole tiles (no requantization).
- Other linear layers: 5 bits, codebook mul1, calibrated through the model with all routed experts at 4 bits (the
  sister repo's file, unchanged).
- The MTP layer: quantized without calibration data (as exllamav3 converts an MTP layer): experts 4 bits mcg, linear
  layers 5 bits mul1 (the sister repo's files, unchanged).
- Tensors exllamav3 keeps unquantized are copied from the BF16 checkpoint as stored (BF16, F32 for the router bias).

## Accuracy

KL divergence of the recipe's output distributions from the original BF16 model's (the official BF16 weights run by
the official implementation, transformers, computed in FP32). Four 8,192-token windows, by position range, measured
with the recipe on four nodes in a deterministic configuration:

| Positions | This checkpoint, FP4 KV cache (as the recipe serves it) | This checkpoint, FP8 KV cache | 4 bits throughout (the sister repo's weights), FP8 KV cache |
|---|---|---|---|
| 1,024 to 2,047 | 0.0190 | 0.0156 | 0.0120 |
| 2,048 to 4,095 | 0.0316 | 0.0258 | 0.0208 |
| 4,096 to 6,143 | 0.0289 | 0.0259 | 0.0198 |
| 6,144 to 8,190 | 0.0192 | 0.0160 | 0.0132 |

The top-1 agreement with the original model is 94.0 to 95.7 % with the FP4 KV cache.

## Speed

On four DGX Sparks with the recipe (release r2-102) and its FP4 KV cache, measured with the GPU clock capped at
2.1 GHz (2,100 MHz) on every node, a coding prompt and one request at a time (median of two runs). Both drafters work:
the MTP layer included here (three drafts a round), or a DFlash2 drafter, which is not included (its model card
states CC BY-NC-ND 4.0, for research and evaluation, with commercial licensing from Inco AI).

| Prompt | Time to first token (MTP / DFlash2) | Decode with the MTP layer | Decode with DFlash2 |
|---|---|---|---|
| 128K (129,440 tokens) | 120.1 s / 121.9 s | 35.3 tokens/s | 42.9 tokens/s |
| 256K (260,229 tokens) | 253.3 s / 256.9 s | 33.2 tokens/s | 40.7 tokens/s |
| 512K (529,884 tokens) | 565.4 s / 571.5 s | 32.2 tokens/s | 37.8 tokens/s |

The recipe's README has the conditions in full: https://github.com/fiore-labo/fiore-glm5.3.

## License

The GLM-5.3 License (`LICENSE`), Copyright (c) 2026 Z.AI, as for the base model: the copyright notice and the permission
notice must be included in all copies or substantial portions. Its clause 2 reads: "If the Licensee or any of its
affiliates operates a Model as a Service business, and the aggregate revenue of the Licensee and its affiliates exceeds
10 billion US dollars (or the equivalent in other currencies) in total over any consecutive 12 months, the Licensee must
pass Z.AI's security review before using the Software or its derivative works for any commercial purpose." Check
whether it applies to you. The recipe's code is Apache-2.0 and licensed separately.
