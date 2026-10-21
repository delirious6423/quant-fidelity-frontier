---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - glm
  - glm-5
  - multimodal
  - vision-language
  - moe
  - sparse-attention
  - mla
  - quantization
  - rfa
  - rfi
  - rocm
  - rdna4
  - gfx1201
  - 4-bit
---

<div align="center">

# GLM-5.3-Flash · RFA + RFI8 composite quant (8× R9700 / RDNA4)

**A self-quantized derivative of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), tuned for 8× AMD Radeon R9700 (gfx1201 / RDNA4)**

[![Model size](https://img.shields.io/badge/params-321B%20(18B%20active)-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)
[![bpw](https://img.shields.io/badge/bpw-4.93-ff69b4)](#quantization)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Serving stack](https://img.shields.io/badge/serving%20stack-tcclaviger%2Fvllm-00a2a2)](https://hub.docker.com/r/tcclaviger/vllm)
[![GPU](https://img.shields.io/badge/GPU-RDNA4%20(gfx1201)-ED1C24)](https://www.amd.com/en/products/processors/desktops/radeon.html)

</div>

---

## Model card

This is a **composite-quantized** checkpoint of Z.ai's **GLM-5.3-Flash** — the first natively
multimodal model in the GLM-5 series (321B total / ~18B active parameters). It was quantized from
the official FP8 release using the `tcclaviger/vllm` composite quantizer and is intended to be
served with **vLLM** on **RDNA4** (AMD Radeon R9700) hardware.

> **📦 Companion serving repo:** [GLM-5.3-Flash-rocm-r9700](https://huggingface.co/djdeniro/GLM-5.3-Flash-rocm-r9700) — the ROCm/RDNA4 overlay + quantization recipe.

### Benchmark

| Configuration | GPQA Diamond | Correct | Empty | Answered |
|---------------|---------|---|---|------|
| GLM-5.3-Flash (Z-AI API) | 80.8% | 76 | 6 | 94 | 
| GLM-5.3-Flash-RFA-RFI8 | 85.1% | 80 | 6 | 94 |

#### Details
 
 - Reasoning Effort: **high**
 - Max Output Len: 32k Tokens


***Total Context Limit for each task in test 32k, means 6x tasks use more than 32k output tokens***

### Serving performance (8× R9700, FY2026-09 production config)

| Scenario | Throughput |
|----------|------------|
| Decode, batch size 1, MTP OFF | ~34–39 tok/s |
| Decode, batch size 1, **MTP spec=3** | **~82–88 tok/s** |
| Aggregate, 4 concurrent, MTP spec=3 | ~155 tok/s |
| Context window (fp8 KV) | 300,000 tokens |

MTP speculative decoding: mean acceptance length ~3.7–3.9 of 4 draft tokens,
average draft acceptance 91–97% (live engine metrics, GPQA-style prompts).


---

## Table of contents

- [Attribution & credits](#attribution--credits)
- [Quantization](#quantization)
- [Quick start](#quick-start)
- [Model details](#model-details)
- [Multimodal policy](#multimodal-policy)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Attribution & credits

| Component | Credit |
|-----------|--------|
| **Base model** | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (Z.ai, MIT) |
| **Quantizer & quant kernels (RFA / RFI)** | [`tcclaviger/vllm:latest`](https://hub.docker.com/r/tcclaviger/vllm) (IronLLM Labs) |
| **RDNA4 port + quant recipe** | [GLM-5.3-Flash-rocm-r9700](https://huggingface.co/djdeniro/GLM-5.3-Flash-rocm-r9700) (this project's overlay) |

The composite **RFA + RFI8** quantization scheme, its kernels, and the serving runtime are provided
by **`tcclaviger/vllm` (IronLLM Labs)**. This checkpoint is the result of applying a quantization
*recipe* (archspec + source patches + kda-remap, in the companion repo) on top of that stack, and a
port of the GLM-5.3-Flash architecture to the RDNA4 serving path.

---

## Quantization

The checkpoint mixes three schemes in one `quant_method: "rfi"` composite:

| Scheme | Bits | Applied to | Stored size |
|--------|------|------------|-------------|
| **RFA** | 4.5 bpw | MoE routed experts (42 layers × 288 experts) | ~171.3 GB |
| **RFI8** | 8 bpw | attention / shared-expert / dense linears | ~7.8 GB |
| **BF16 / FP32** | 16 / 32 bpw | embeddings, vision tower, norms, MTP layer | ~18.7 GB |

### Average bits-per-weight

```
bpw = (total safetensors bytes × 8) / total parameters
    = (197,843,715,288 × 8) / 321,342,220,638
    = 4.9254  ≈  4.93 bpw
```

| Metric | Value |
|--------|-------|
| Total parameters | 321,342,220,638 (~321.3B) |
| On-disk size | 197.8 GB · 25 safetensors shards |
| **Average bpw** | **4.9254 ≈ 4.93** |
| vs. FP8 source | **0.60×** (197.8 GB vs 328.3 GB) |
| vs. BF16 | **0.31×** (197.8 GB vs 642.7 GB) |

---

## Quick start

```bash
docker pull tcclaviger/vllm:latest

git clone https://huggingface.co/djdeniro/GLM-5.3-Flash-rocm-r9700 overlay

# current production config (MTP spec=3, fp8 KV, 300k context)
docker run --rm --tty --ipc=host --shm-size=128g \
  --device /dev/kfd:/dev/kfd --device /dev/dri:/dev/dri \
  -v /path/to/GLM-5.3-Flash-RFA-RFI8-8xR9700:/models:ro \
  -v "$PWD/overlay":/overlay:ro \
  --entrypoint bash tcclaviger/vllm:latest \
  -c "/overlay/apply_overlay.sh && GLM5_NEXT_MTP_PROPOSER=1 exec vllm serve /models \
      --served-model-name glm53-flash --trust-remote-code --quantization rfi \
      --tensor-parallel-size 8 --gpu-memory-utilization 0.9575 \
      --max-model-len 300000 --max-num-seqs 4 --max-num-batched-tokens 2048 \
      --kv-cache-dtype fp8 \
      --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
      --enable-prefix-caching --distributed-executor-backend mp \
      --compilation-config '{\"cudagraph_capture_sizes\":[1,2,4,8,16],\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_copy_inputs\":true}'"
```

---

## Model details

| Property | Value |
|----------|-------|
| Architecture | `Glm5NextForConditionalGeneration` |
| Layers | 45 = 34 KDA (linear attention) + 11 DSA (sparse-MLA) |
| Routed experts | 288 (top-8) + 1 shared expert |
| Extra | mHC hyper-connections, 1 nextn MTP draft layer, native vision tower |
| Context (fp8 KV) | 300,000 tokens |

---

## Multimodal policy

Images are resized with aspect ratio preserved, clamped to **min 384×384 / max 1280×1280**, and
fed with a min/max image-token budget. The model accepts image and video inputs natively.

---

## Known limitations

- **fp8 KV without runtime calibration** — serve with `--kv-cache-dtype fp8` and **scales fixed
  at 1.0**. Do not enable `--calculate-kv-scales`: runtime calibration on the profile dummy-run
  produces garbage scales from the uninitialized KDA recurrent state (details in the
  [serving repo](https://huggingface.co/djdeniro/GLM-5.3-Flash-rocm-r9700)).
- The 300k context / MTP spec=3 config presumes the VRAM headroom of the 256 GB 8× R9700 node.
- **Chat needs `reasoning_effort="low"`** — the default Reasoning Effort Max spends 16k+ tokens
  thinking before producing content on long generations.

---

## License

MIT. Base model © Z.ai (zai-org), MIT license. Quantizer & runtime © IronLLM Labs
([tcclaviger/vllm](https://hub.docker.com/r/tcclaviger/vllm)).
