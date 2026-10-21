---
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
language: [en, zh]
---

# DeepSeek-V4-Flash-0731 - IQ2XXS 2-bit GGUF

Quantized from the official `deepseek-ai/DeepSeek-V4-Flash-0731` safetensors (48 shards, FP4/FP8) with the [antirez/ds4](https://github.com/antirez/ds4) `gguf-tools/deepseek4-quantize` pipeline.

## Recipe: IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8

| Tensor family | Type |
|---|---|
| Routed experts (gate/up) | `iq2_xxs` |
| Routed experts (down) | `q2_K` |
| Attention projections | `q8_0` |
| Shared experts | `q8_0` |
| Output head | `q8_0` |
| Token embeddings | `f16` |
| Compressor / Indexer / Hyper-connections | `f16` |

- **Size:** 86.5 GB (1328 tensors)
- **Architecture:** `deepseek4`, 43 layers, 256 routed experts (6 used), hash routing (3 layers), compressed KV + indexer, hyper-connections
- **Context:** 1,048,576 tokens
- **Quantization:** GGUF v3, imatrix: weight-energy fallback (V2 with activation imatrix planned)
- tid2eid stored as I32 (ds4-engine compatible)

## Usage

Runs with the `ds4` engine (antirez/ds4, supports DeepSeek-V4-Flash-0731):

```
./ds4 -m DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf --ctx 32768
```

Combine with `DeepSeek-V4-Flash-0731-DSpark-support.gguf` (sm54) for DSpark speculative decoding:

```
./ds4 -m ... -mtp DeepSeek-V4-Flash-0731-DSpark-support.gguf --dspark
```

See the source model repo for tokenizer/config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
