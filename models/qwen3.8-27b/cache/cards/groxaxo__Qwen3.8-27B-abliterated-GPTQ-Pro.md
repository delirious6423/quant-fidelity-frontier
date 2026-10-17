---
license: apache-2.0
base_model:
- huihui-ai/Huihui-Qwen3.8-27B-abliterated
- Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3.8
- qwen3.5-architecture
- abliterated
- gptq-pro
- gptq
- 4-bit
- 4bit
- uncensored
- mtp
- speculative-decoding
- text-generation
- conversational
---

# Qwen3.8-27B-abliterated-GPTQ-Pro

## MTP performance

Measured with vLLM on 2× RTX 3090 GPUs using tensor parallelism, a 4K context, greedy decoding,
and 256 generated tokens.

| Configuration | Single-stream tok/s | Aggregate tok/s (8 concurrent) |
|---|---:|---:|
| No MTP | 71.5 | 390.6 / 430.9 |
| **MTP (`num_speculative_tokens=1`)** | **105.5 (+47.6%)** | **426.4 / 555.4 (+9% / +28.9%)** |

<!-- polished-overview:start -->
## Overview

**Qwen3.8-27B-abliterated-GPTQ-Pro** is a GPTQ-quantized checkpoint intended for efficient GPU inference, published by [`groxaxo`](https://huggingface.co/groxaxo).
It is intended for open-source evaluation, reproducible experimentation, and compatible local or
hosted inference workflows. The wording below is deliberately limited to what can be verified
from this repository's metadata and artifacts.


The repository name identifies a behavior-modified or reduced-filtering lineage. That label describes the source or conversion history; it is not a guarantee of unrestricted behavior in every prompt or runtime. Test outputs carefully before sharing or deploying them.

## At a glance

| Field | Details |
|---|---|
| Format | **GPTQ** |
| Source / base | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Intended task | `image-text-to-text` |
| License | `apache-2.0` |

## What is included

- `*.safetensors` (5 files)
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `chat_template.jinja`
- `quantize_config.json`
- Additional configuration, tokenizer, processor, or shard files (18 visible artifacts total)

## Quick start

### vLLM (documented configuration)

```bash
vllm serve groxaxo/Qwen3.8-27B-abliterated-GPTQ-Pro \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --dtype bfloat16 --language-model-only \
  --max-model-len 260000 --gpu-memory-utilization 0.92 --max-num-seqs 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

This command is taken from the repository documentation. Adjust tensor parallelism, context
length, and cache settings to match your hardware and vLLM version.

## Compatibility and responsible use

- Use a runtime that explicitly supports this format, architecture, and modality.
- Keep configuration, tokenizer, processor, projection, and weight files from the same revision together.
- Review the source model card and license before redistribution or deployment.
- Hardware needs depend on parameter count, context length, cache precision, quantization, and concurrency.
- Report reproducible issues with the runtime version, hardware, launch command, and a minimal example.

Quantization or conversion changes numerical behavior, memory use, and throughput relative to the source checkpoint; validate quality on your own workload.

Generated outputs may be inaccurate or unsuitable for a given use case. Users are responsible for
testing behavior, applying appropriate safeguards, and complying with applicable licenses and laws.
<!-- polished-overview:end -->

INT4 GPTQ-Pro quantization of [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), with the **MTP (multi-token prediction) draft head preserved and served natively** by vLLM's speculative decoding.

This is the **calib-128** build, using 128 calibration samples.

## Quantization details

| Parameter | Value |
|---|---|
| Tool | GPTQModel `6.1.0-dev` (GPTQ-Pro) |
| Bits / symmetry | 4-bit, symmetric |
| Group size | 64 (`desc_act=False`) |
| Calibration | text mode, 128 samples |
| Preset | `quality` (true-sequential, activation-weighted MSE, group-aware) |
| Coverage | 400/400 linear modules across all 64 layers (48 linear-attention GDN + 16 full-attention) |
| MTP head | preserved unquantized (`mtp.*` tensors kept; `mtp_num_hidden_layers=1`) |
| Offload | disk offload during calibration |

Per-module quantization loss (`quant_log.csv`, included): mean **8.2e-5**, max **4.9e-4**. Damping 1.49–8.94 (mean 3.33).

## vLLM MTP speed test (measured, 2× RTX 3090, TP=2)

vLLM 0.24.0, `max_model_len=4096`, greedy, 256 generated tokens, prompt about MTP itself. Greedy decoding → measured draft acceptance at position 0: **100%**.

| Config | Single-stream tok/s | Aggregate tok/s (8 concurrent) |
|---|---|---|
| No MTP | 71.5 | 390.6 / 430.9 |
| **MTP (`num_speculative_tokens=1`)** | **105.5 (+47.6%)** | **426.4 / 555.4 (+9% / +28.9%)** |

Single-GPU note: on one RTX 3090 (23.56 GiB) the MTP drafter does **not** fit alongside this checkpoint — target ~21.5 GiB + drafter ~2.4 GiB exceeds the card. Single-GPU no-MTP baseline for reference: 42.9 tok/s single-stream, ~273 tok/s at 8-way concurrency.

## Production serving (2× RTX 3090, TP=2, vLLM 0.24.0)

Deployed behind a router serving real chat traffic (temp 0.7):

| Setting | Value |
|---|---|
| Tensor parallel | 2 (21.6 GiB/GPU at `gpu_memory_utilization=0.92`) |
| `max_model_len` | 260,000 |
| KV cache dtype | bfloat16 (FP8 KV tried and dropped) |
| `max_num_seqs` | 2 |
| MTP | draft-3 (`num_speculative_tokens=3`) |
| KV pool | 331,750 tokens — one full 260K sequence + ~72K spare |
| Cold start | ~150 s (compile-cached); idle reap 300 s |

**Measured single-stream speeds (temp 0.7 unless noted):**

| Config | tok/s |
|---|---|
| No MTP (4K bench ctx) | 71.5 |
| MTP draft-1, greedy (4K bench ctx) | 105.5 |
| MTP draft-3, 262K ctx, FP8 KV | 78–80 |
| **MTP draft-3, 260K ctx, bf16 KV (final)** | **84–87** |
| MTP draft-1 at temp 0.7 | 66–71 |

Findings:

- **Use draft-3, not draft-1.** Draft-1 wins only under greedy decoding (100% position-0 acceptance); at real sampling temps it is *slower* (~77% acceptance → ~1.77 tok/step) than draft-3 (~2.76 tok/step).
- **Speculative decoding degrades under batching**: 2 concurrent streams yield 56.9 tok/s *combined* vs 84 sequential. `max_num_seqs=1` maximizes single-stream speed; `2` trades aggregate throughput for simultaneous admission.
- bf16 KV halves KV pool vs FP8 (331,750 tokens) but no measurable quality/speed penalty in chat use.
- `min_p` and `logit_bias` do not work with speculative decoding (vLLM limitation).

**⚠ Critical flag for multi-GPU RTX 3090 (PCIe, no NVLink): pass `--disable-custom-all-reduce`.** vLLM's custom allreduce kernel silently corrupts tensor-parallel output on this hardware class — endless `!!!!...` with zero errors. Verified by A/B: identical checkpoint and prompts, broken without the flag, byte-identical correct output with it. `NCCL_P2P_DISABLE=1` does **not** fix this. Affects any TP-served model on such boxes, not just this quant.

## Usage

```bash
vllm serve groxaxo/Qwen3.8-27B-abliterated-GPTQ-Pro \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --dtype bfloat16 --language-model-only \
  --max-model-len 260000 --gpu-memory-utilization 0.92 --max-num-seqs 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Without MTP, serve like any GPTQ checkpoint (single 24 GB GPU works with `--max-model-len 4096 --gpu-memory-utilization 0.92 --max-num-seqs 32`).

Note: `min_p` and `logit_bias` do not work with speculative decoding (vLLM warning).

## Acknowledgements

- [huihui-ai](https://huggingface.co/huihui-ai) for the abliterated source model
- [Qwen](https://huggingface.co/Qwen) — Qwen3.8-27B
- [GPTQModel / ModelCloud](https://github.com/modelcloud/gptqmodel) for GPTQ-Pro
