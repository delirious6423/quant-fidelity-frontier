---
quantized_by: LemesoftNostalgic
pipeline_tag: image-text-to-text
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
---

## Llamacpp imatrix IQ4_XS Quantization of Ornith-1.5-35B-A3B by ornith-ai

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> for quantization.

Original model: https://huggingface.co/Ornith-ai/Ornith-1.5-35B-A3B

**Model details:**
- Parameter count: 35B
- Input support: text, image (with mmproj file) - [details](#multimodal)
- Speculative decoding: yes (MTP) - [details](#mtp)
- Perplexity check: yes - [details](#perplexity)

## Overview

The sole reason for creating this model: the excellent GGUF model array of ornith-ai is missing a model that fits in 24GB VRAM budget with reasonable KV-cache. One example of such a setup is a budget rig with 2 x RTX 3060 that I happen to possess. I approached the problem by choosing the output tensor types and token embedding types the same as Unsloth's great Qwen3.6-35B-A3B-UD-IQ4_XS.gguf, creating the imatrix with Qwen3.8-27B-calibration-v6.txt from bartkowski and quantizing with the IQ4_XS profile. The base model I used: ornith-ai/Ornith-1.5-35B-BF16.gguf


## Available files:

| Filename | Quant type | File Size | Split | bpw |
| -------- | ---------- | --------- | ----- | --- |
| [Ornith-1.5-35B-IQ4_XS.gguf](https://huggingface.co/LemesoftNostalgic/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-IQ4_XS.gguf) | IQ4_XS | 18.2 GB | false | 4.24 |

## Running example

This quant runs with [llama.cpp](https://github.com/ggml-org/llama.cpp). Running example for 2 x RTX 3060, 2 x PCIe Gen3 x16, 24-thread CPU e.g. Xeon E5-2690v3, and lightweight OS such as XUbuntu:

```bash
llama-server --jinja \
  -m Ornith-1.5-35B-IQ4_XS.gguf \
  -kvo \
  -t 10 \
  -tb 24 \
  -ngl 999 \
  --cache-type-k q8_0 \
  --cache-type-v q5_0 \
  -fa on \
  -fit off \
  -c 180224 \
  -sm tensor \
  --gpu-layers-draft all \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --port 8080 \
  --reasoning-effort medium \
  -np 1 \
  --load-mode none \
  --no-warmup \
  -b 256 \
  -ub 128 \
  --spec-draft-type-k q8_0 \
  --spec-draft-type-v q8_0 \
  --mmproj mmproj-Ornith-1.5-35B-BF16.gguf
```

With 2 x RTX 3060 the peak fill rate is about 1800 tok/s and peak generation rate is about 120 tok/s. With agentic use, a lot of data in the KV-cache, the speed hovers at 50-90 tok/s. This thing is fast, and it can code straightforward tasks quite well.

## Multimodal

This model supports image input. The multimodal projector file from the base GGUF repo is used.

## MTP

This model has MTP (Multi-Token Prediction) layers.

## Perplexity

A quick perplexity check against the bf16 model on `wiki.test.raw` over 406 lines. Cor(ln(PPL(Q)), ln(PPL(base)): 98.41%. As a reference, the official Q4_K_M image has 98.58%, and all the other numbers very close the same.

## Credits

Thanks for Unsloth for the inspiration e.g. its excellent UD-IQ4_XS model types.

Thanks for bartkowski for the Qwen3.8-27B-calibration-v6.txt used in the imatrix creation.
