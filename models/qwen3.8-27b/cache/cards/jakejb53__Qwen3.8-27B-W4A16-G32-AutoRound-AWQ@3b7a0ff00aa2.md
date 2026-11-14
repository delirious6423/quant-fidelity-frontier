---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- qwen3_5
- conversational
- qwen3.8
- W4A16
- G32
- AutoRound
- AWQ
---
## **Qwen3.8-27B** · **AutoRound W4A16 G32** · **AWQ GEMM**

Goal: Maximum accuracy full 262K context with 4 concurrency on 2xRTX 3090 with vLLM.

4-bit asymmetric weight-only quantization using `AutoRound 0.17.0-dev` with `SignRoundV2` tuning and `NeUQI` initialization. `AWQ` `GEMM` packed, with 32-weight groups and FP16 scales.
4.6 bpw quantized · 6.2 effective bits/parameter · 600 tuning iterations · 512 calibration samples
Vision tower, MTP head, embeddings, output head, sensitive linear-attention gates, norms, and `Conv1D` components retained in BF16.

#### Base model

| Attribute | Value | |
|---|---|---|
| Publisher | Qwen | `Qwen` |
| Family and size | Qwen3.8, 27B | `Qwen3.8-27B` |
| Modality | image and video plus text | `VL`, `VLM`, `Vision` |
| Architecture | hybrid: 48 Gated DeltaNet layers, 16 full-attention | `GDN`, `Hybrid` |
| Weights | stock, not fine-tuned or modified | `base`, `stock` |
| Source precision | BF16 | `BF16` |
| Native context | 262,144 tokens | `262K` |

#### Quantization scheme

| Attribute | Value | |
|---|---|---|
| Weight bits | 4 | `W4`, `4bit`, `INT4`, `int4` |
| Activation bits | 16 (unquantized) | `A16`, `W4A16` |
| Scope | weight-only | `WOQ` |
| Data type | integer | `INT`, `int` |
| Symmetry | asymmetric, per-group zero point | `asym`, `zp` |
| Group size | 32 | `G32`, `g32`, `gs32` |
| Scale precision | fp16 | `fp16-scale` |
| Bit allocation | uniform 4-bit on all quantized layers, not mixed-bit | `uniform` |
| Bits per quantized weight | ~4.6 (computed) | `4.6bpw` |
| Bits per parameter, whole file | ~6.2 (computed, approximate) | `6.2bpw` |

#### Method

| Attribute | Value | |
|---|---|---|
| Toolkit | AutoRound 0.17.0 (dev build) | `AutoRound`, `AR`, `AR0.17` |
| Algorithm | SignRound V2 | `SignRoundV2`, `ARv2`, `v2` |
| Starting-point search | NeUQI | `NeUQI` |
| Tuned or not | tuned, not round-to-nearest | `tuned` |
| Iterations per block | 600 | `i600`, `600it` |
| Calibration samples | 512 | `n512` |
| Calibration length | 2,048 tokens | `s2048` |
| Batch size, seed | 8, 42 | |

#### Calibration data

| Attribute | Value | |
|---|---|---|
| Sources | UltraChat 50%, OpenCodeInstruct 37.5%, pile-10k 12.5% | `chat-code`, `mix` |
| Rendering | chat-templated, no system turn | `templated` |
| Reasoning effort in rendering | medium | `medium`, `med` |
| Language | English | `en` |
| Modality | text only, no images | `text-calib` |

#### Format and runtime

| Attribute | Value | |
|---|---|---|
| Checkpoint format | AWQ layout (`quant_method: awq`, `gemm`) | `AWQ` |
| Container | safetensors, 4 shards plus extras file | |
| Intended engine | vLLM | `vLLM` |
| Chat template | stock, unmodified | |

#### What was left at full precision

| Attribute | Value | |
|---|---|---|
| Vision tower | `BF16-vision`, `vision-fp` |
| MTP head (kept, usable for speculative decoding) | `MTP`, `MTP-BF16` |
| `lm_head` and embeddings | `fp-head` |
| Linear-attention `in_proj_a/b` gates | `GDN-gates-BF16` |

#### Size and target

| Attribute | Value | |
|---|---|---|
| On disk | 19.5GiB (21.0GB) | `20GB`, `21GB` |
| Relative to BF16 | about 39%, or 2.6× smaller (computed) | |
| Target hardware | 2 × RTX 3090, 48GB, TP=2 | `2x3090`, `48GB`, `TP2` |
| Target context and KV cache | 262,144 tokens, BF16 KV | `262K`, `BF16-KV` |