---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - qwen3.8
  - compressed-tensors
  - w8a16
  - int8
  - quantized
  - vllm
  - dflash2
  - speculative-decoding
  - vision
  - conversational
base_model_relation: quantized
---

<div align="center">
  <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png" width="380" alt="Qwen">
  <h1>Qwen3.8-27B · INT8 W8A16 · DFlash2</h1>
  <p><strong>Fast, near-lossless Qwen3.8-27B for dual RTX 3090s. Full 262K context.</strong></p>
  <p>
    <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Base model</a> ·
    <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a> ·
    <a href="https://github.com/vllm-project/vllm">vLLM</a> ·
    <a href="https://github.com/vllm-project/llm-compressor">llm-compressor</a>
  </p>
  <p>
    <img alt="Base Qwen3.8-27B" src="https://img.shields.io/badge/base-Qwen3.8--27B-8B5CF6">
    <img alt="Format W8A16" src="https://img.shields.io/badge/format-W8A16-0A2EFE">
    <img alt="Weights INT8" src="https://img.shields.io/badge/weights-INT8-6C4CF1">
    <img alt="Activations FP16 or BF16" src="https://img.shields.io/badge/activations-FP16%20%2F%20BF16-555555">
    <img alt="Target Ampere" src="https://img.shields.io/badge/target-Ampere%20sm__86-76B900">
    <img alt="License Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue">
  </p>
</div>

> [!NOTE]
> A numerical INT8 W8A16 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), served with the DFlash2 drafter for speculative decoding. All model credit belongs to Qwen; this repository changes numerics only. The detailed engineering notes live in [TECHNICAL.md](TECHNICAL.md).

## The short version

Qwen3.8-27B is a big model. This build makes it fast on consumer hardware:

- **Up to 117 t/s decode** on two RTX 3090s at full 262K context
- **~1.4x to 1.5x faster** than the same model with native MTP4 speculation
- **~2.5x faster** than plain autoregressive decoding
- **98% top-1 agreement** with the original BF16 model (mean KLD 0.0007)
- **28 GiB** on disk, **2.02 GiB** drafter, full 262K context on 2×24 GB

## How fast

Measured on two RTX 3090s: same engine, same 262K context, same prompts, single stream, cold cache. Generation tokens per second.

| Prompt tokens | Autoregressive | MTP4 | **DFlash2** | DFlash2 vs MTP4 |
|---:|---:|---:|---:|---:|
| 128 | 47 | 77 | **117** | **1.5x** |
| 2,048 | 47 | 73 | **103** | **1.4x** |
| 8,192 | 47 | 75 | **102** | **1.4x** |

DFlash2 wins at every prompt length we tested. The reason is the drafter: DFlash2 proposes 7 draft tokens per step and the target verifies them all in one pass. Native MTP proposes 4. More drafts per verification step means more accepted tokens and less idle GPU time.

## How close to the original

| | BF16 original | This build |
|---|---:|---:|
| Size | ~56 GiB | **28 GiB** |
| Mean KLD | 0 | **0.0007** |
| Top-1 agreement | 100% | **98.1%** |

KLD is a fancy way of asking "does the quantized model pick the same next token as the original?" Lower is closer. A mean of 0.0007 nats means the INT8 weights and the BF16 weights are nearly interchangeable. Halving the size is what lets the full 262K context fit on two 3090s.

## What you get

- Full 262,144-token context on 2×RTX 3090
- Vision tower, thinking controls, and tool calling intact, same as upstream Qwen3.8-27B
- Native MTP removed: DFlash2 replaces it, so there is no dead weight in the repo
- Runs on vLLM with the [club-3090](https://github.com/noonghunna/club-3090) patch set (DFlash2 is a new spec-decoder, still in PR review upstream)

## Run it

Three steps. The full command with every flag is in [TECHNICAL.md](TECHNICAL.md).

```bash
# 1. Models (pulls into ~/.cache/huggingface)
hf download lued/Qwen3.8-27B-INT8-W8A16-DFlash2
hf download lued/Qwen3.8-27B-DFlash2-W8

# 2. Patches (DFlash2 support is not in a released vLLM yet)
git clone https://github.com/noonghunna/club-3090.git

# 3. Serve (condensed; full command in TECHNICAL.md)
export CLUB3090="$HOME/club-3090"
podman run --rm --replace --device nvidia.com/gpu=all --ipc=host -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$CLUB3090/models/qwen3.6-27b/vllm/patches/vllm-pr52816-dflash2":/etc/club3090/pr52816:ro \
  -v "$CLUB3090/models/qwen3.6-27b/vllm/patches/vllm-pr48375-mamba-drop-eagle-block":/etc/club3090/pr48375:ro \
  --entrypoint bash docker.io/vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d \
  -c 'bash /etc/club3090/pr48375/install.sh || exit 1; bash /etc/club3090/pr52816/install.sh || exit 1; exec vllm serve "$@"' -- \
  lued/Qwen3.8-27B-INT8-W8A16-DFlash2 \
  --tensor-parallel-size 2 --max-model-len 262144 --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"dflash","model":"lued/Qwen3.8-27B-DFlash2-W8","num_speculative_tokens":7}'
```

## Status

Built, audited, and published 2026-08-19. Both release audits and the drafter audit pass, KLD measured, serving validated at 262K on two 3090s. The drafter ships separately as [lued/Qwen3.8-27B-DFlash2-W8](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8). All engineering detail, audit evidence, and measured tables are in [TECHNICAL.md](TECHNICAL.md).
