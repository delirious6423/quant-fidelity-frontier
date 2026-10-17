---
language:
  - en
  - zh
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - gguf
  - llama.cpp
  - qwen
  - qwen3
  - qwen3.8
  - mtp
  - speculative-decoding
  - yarn
  - long-context
  - 1m-context
  - imatrix
  - quantized
  - agent
  - react
pipeline_tag: image-text-to-text
license: apache-2.0
library_name: gguf
inference:
  parameters:
    temperature: 1.0
    top_p: 0.95
---

# Qwen3.8-27B-Imatrix-Mtp-1m

Right after Qwen3.8 was open-sourced, we tested it immediately: instruction following and debugging/troubleshooting capabilities are both greatly improved. However, the open release is natively capped at 262,144 tokens — the 1M context requires extension. Following the **official YaRN recipe**, we built a 1M version. This repo is that build, in GGUF format, with a verified-working MTP head.

Two changes vs. the base release:

1. **1M context** — extended from the native 262,144 tokens using the official YaRN recipe (see [1M context (YaRN)](#1m-context-yarn)).
2. **Working MTP head** — the Multi-Token Prediction head (`blk.64`) is included in every file and **verified to work** with speculative decoding in llama.cpp / LM Studio. Acceptance rates are on par with the official unsloth GGUFs.

## Available files

| File | Quantization | MTP | Context |
|---|---|---|---|
| `Qwen3.8-27B-Imatrix-Mtp-1m-F16.gguf` | F16 (lossless) | yes | 1M |
| `Qwen3.8-27B-Imatrix-Mtp-1m-Q8_0.gguf` | Q8_0 | yes | 1M |
| `Qwen3.8-27B-Imatrix-Mtp-1m-Q6_K.gguf` | Q6_K | yes | 1M |
| `Qwen3.8-27B-Imatrix-Mtp-1m-Q5_K_M.gguf` | Q5_K_M | yes | 1M |
| `Qwen3.8-27B-Imatrix-Mtp-1m-Q4_K_M.gguf` | Q4_K_M (imatrix) | yes | 1M |
| `Qwen3.8-27B-Imatrix-Mtp-1m-Q3_K_M.gguf` | Q3_K_M | yes | 1M |

All files are converted from the official HF weights; non-MTP tensors are bit-identical to the source (modulo quantization).

## Requirements

- A llama.cpp build with `draft-mtp` speculative support (verified on **b10252 / `fe2adf0e7`**, CUDA 12, RTX PRO 6000 Blackwell).
- **LM Studio 2.28.2** (engine 2.28.2 / b10252) verified.
- For 1M context you need a lot of VRAM for the KV cache — use quantized KV cache (`q8_0` recommended).

## Usage

### llama.cpp

```shell
llama-server -m Qwen3.8-27B-Imatrix-Mtp-1m-Q4_K_M.gguf \
  --ctx-size 262144 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-n-max 2
```

For full 1M context, raise `--ctx-size` to `1000000` (ensure enough VRAM / use `--flash-attn` and quantized KV cache).

### LM Studio

Load the GGUF, set context length, and enable MTP (draft max tokens = 2). No extra configuration is needed — the MTP head is embedded in the file.

## Verified MTP performance

Test hardware: **NVIDIA RTX PRO 6000 Blackwell** · llama.cpp b10252 (`fe2adf0e7`), CUDA 12 · LM Studio 2.28.2 (same engine).

Measured on the same engine, same prompt, `temp=0.7`, 400-token generation (Q4_K_M):

| Model | Draft acceptance |
|---|---|
| Official unsloth Q4_K_M | 55.7% / 56.4% |
| This build (Q4_K_M) | 50.8% / 56.1% |

Up to ~77% acceptance on easy prompts (`temp=0`). Speedup vs. MTP disabled (Q4_K_M, median of 3 runs):

| Context | MTP off | MTP on | Speedup |
|---|---|---|---|
| 256 tok | 63.7 tok/s | 79.6 | +25% |
| 2K | 58.9 | 94.6 | +61% |
| 8K | 58.1 | 95.7 | +65% |
| 16K | 56.4 | 97.6 | +73% |
| 32K | 54.5 | 92.8 | +70% |
| 64K | 49.7 | 89.8 | +81% |

The speedup grows with context length, and MTP-on throughput stays flat where plain decode degrades.

## Technical note: MTP head RMSNorm convention

If you convert Qwen3.5/3.6/3.8 weights to GGUF yourself, watch out for this: the HF `Qwen3_5RMSNorm` is defined as

```python
self.weight = nn.Parameter(torch.zeros(dim))
output = output * (1.0 + self.weight.float())
```

i.e. HF stores the **offset relative to 1.0**, while llama.cpp applies `rms_norm(x) * w_gguf` directly. Therefore GGUF must store `1.0 + HF_weight`.

The MTP head (`blk.64`) has 7 such norm tensors that are easy to miss in the tensor map:

```
blk.64.attn_norm.weight
blk.64.attn_q_norm.weight
blk.64.attn_k_norm.weight
blk.64.post_attention_norm.weight
blk.64.nextn.enorm.weight
blk.64.nextn.hnorm.weight
blk.64.nextn.shared_head_norm.weight
```

If they are stored without the `+1.0`, the MTP head produces garbage drafts and speculative acceptance drops to **0%** (symptom: draft tokens are mid-word fragments with p ≈ 0.1). All files in this repo have the correct values.

## 1M context (YaRN)

Following the official recipe, the rope parameters are set to:

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

Note: this is **static YaRN** — the scaling factor is constant regardless of input length, which can slightly affect short-text performance. If you only need short contexts, prefer the official release.

## Sampling parameters

From the official model card:

- **Thinking mode** (default): `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
- **Instruct / non-thinking mode**: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`

For long-horizon agentic tasks, allocate generous output length (e.g. 262,144 for reasoning + 131,072 for the final response within the 1M window).

## License

Apache-2.0. Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen Team.
