---
license: mit
language:
  - en
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
  - text-generation
  - moe
  - gguf
  - qwen35moe
  - mtp
  - speculative-decoding
model_creator: configur8tor
model_name: Ornith-1.5-35B-A3B (APEX-MTP-Fixed)
---

# Ornith-1.5-35B-A3B — APEX-MTP-Fixed

**Repaired multi-token-prediction (MTP) head build of Ornith-1.5-35B-A3B, exported as GGUF.**

The stock Ornith-1.5 MTP head shipped **untrained** — its RMSNorm scales read garbage (~1e31 / NaN) — so speculative decoding was dead on arrival. This build splices in the **Avifenesh continued-trained MTP head** (dequantized NVFP4 → f32 → Q8_0), restoring a properly-trained head so in-model MTP speculative decoding actually works.

## Verified model facts (read from this exact GGUF)
| Property | Value |
|---|---|
| Architecture | `qwen35moe` |
| Parameters | **35.505B total** (256 experts, 8 active, ~2.6B active per token) |
| Context length | **262,144** (256k) |
| Decoder blocks | 41 |
| MTP | `nextn_predict_layers = 1` (MTP spec-decode enabled) |
| Embedding dim | 2048, RoPE base 1e7, full-attention every 4th layer, SSM layers |

## Quantization (verified from the blob)
| Component | GGML type | Count |
|---|---|---|
| Routed experts | Q4_K | 221 |
| Higher-precision layers | Q6_K | 121 |
| Deepest-compressed experts | Q3_K | 90 |
| **MTP head (`blk.40`)** | **Q8_0** | **11** |
| MTP head norms | F32 | 9 |
| Norms / embeddings | F32 | 310 |
| **File size** | **17.44 GB** (17,437,861,152 B) |

## The MTP head fix
- **Original head:** untrained — RMSNorm scales read garbage (~1e31, NaN), so spec-decode output was nonsense.
- **Avifenesh continued-trained head:** RMSNorm scales 0.97–1.05 (a properly trained head).
- **Effect:** fixes depth-2/3 prediction collapse (0.80/0.27/0.13 → 0.81/0.58/0.43) and lifts draft acceptance ≈ 0.352 → 0.431.
- **Splice integrity:** NVFP4 → f32 → Q8_0 round-trip, MSE 0.000000 (lossless).

## Usage
Runs on llama.cpp and Ollama (both honor `qwen35moe` MTP via `nextn_predict_layers=1`). Fast on a 4070-class GPU.

```bash
# Ollama (also published there)
ollama run slickwillies/ornith15-35b-a3b-apex-mtp-fixed
```

## Acknowledgements
- Base: `ornith-ai/Ornith-1.5-35B-A3B`
- Trained MTP head: Avifenesh continued-training
- GGUF base: `mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF` (APEX-MTP-Compact)
- Splice + repack: configur8tor
