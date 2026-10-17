---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- qwen
- qwen3.8
- flash
- cq
- cq_quant
- quantization
- 4bit
- vllm
- text-generation
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next-CQ-4bit

> **Qwen3.8-Flash-Next · Coded Quantization (CQ) 4-bit**
>
> **Note on Hugging Face's model-size display:** Hugging Face may report a smaller stored parameter count for this repository because CQ stores model weights in a packed 4-bit representation. The displayed stored tensor count therefore does not directly represent the parameter count of the original model architecture.

**Qwen3.8-Flash-Next quantized to 4-bit with Coded Quantization (CQ).**

**ThousandML:** https://thousandml.com

CQ is a **calibration-free 4-bit quantization method** designed to preserve the behavior of the original model without requiring a calibration dataset. Because CQ does not optimize the quantization around a fixed calibration corpus, it is designed to generalize beyond calibration-specific data and workloads.

This repository contains **Qwen3.8-Flash-Next** quantized with **Coded Quantization (CQ) 4-bit**.

## Why CQ?

**Calibration-free.** CQ does not require a representative calibration dataset before quantization.

This avoids tying the quantization procedure to a particular calibration corpus and is intended to provide better generalization beyond the data that would otherwise have been used for calibration.

**4-bit inference.** Model weights are quantized to 4-bit for substantially lower weight memory requirements than BF16/FP16 deployment.

**Native vLLM integration.** CQ models run through the CQ vLLM quantization plugin and custom CUDA kernels included in the Thousandml CQ Docker image.

## Model

| | |
|---|---|
| **Base model** | `Qwen/Qwen3.8-Flash-Next` |
| **Quantization** | **Coded Quantization (CQ) 4-bit** |
| **Calibration** | **None — calibration-free** |
| **Runtime** | vLLM + CQ custom kernels |
| **Quantization name** | `cq` |
| **Context length** | **262,144 tokens** |

## Run with Docker

The recommended way to run this model is with the dedicated Qwen3.8-Flash-Next CQ Docker image.

This image contains the compatible vLLM installation, CQ quantization plugin, CQ CUDA kernels, and runtime dependencies required by Qwen3.8-Flash-Next.

### 1. Make sure NVIDIA Docker support works

```bash id="dmv9wn"
docker run --rm --gpus all \
  nvidia/cuda:13.0.2-base-ubuntu22.04 \
  nvidia-smi
```

Your GPU should appear in the output.

### 2. Start Qwen3.8-Flash-Next-CQ-4bit

```bash id="9hww4w"
docker run --rm --gpus all \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-qwen38-flash-next-v1 \
  vllm serve thousandml/Qwen3.8-Flash-Next-CQ-4bit \
  --quantization cq
```

The model is then available through vLLM's OpenAI-compatible API on port `8000`.

### WSL2

If you are running Docker through WSL2 and vLLM reports that UVA is unavailable, enable vLLM's WSL2 pinned-memory support:

```bash id="zfrr1v"
docker run --rm --gpus all \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-qwen38-flash-next-v1 \
  vllm serve thousandml/Qwen3.8-Flash-Next-CQ-4bit \
  --quantization cq
```

## Send a request

Once the server is ready:

```bash id="fctd77"
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "thousandml/Qwen3.8-Flash-Next-CQ-4bit",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantization in one paragraph."
      }
    ]
  }'
```

## Reasoning and tool calling

Qwen3.8-Flash-Next can be served with vLLM's reasoning and automatic tool-choice support.

```bash id="r4dfz8"
docker run --rm --gpus all \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-qwen38-flash-next-v1 \
  vllm serve thousandml/Qwen3.8-Flash-Next-CQ-4bit \
  --quantization cq \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

## Hugging Face authentication

For higher Hugging Face Hub rate limits, pass your Hugging Face token into the container:

```bash id="1owhhr"
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-qwen38-flash-next-v1 \
  vllm serve thousandml/Qwen3.8-Flash-Next-CQ-4bit \
  --quantization cq
```

On WSL2, combine it with:

```bash id="qf1spz"
-e VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

## Full recommended command

For deployments using reasoning, automatic tool calling, and the full supported context length:

```bash id="vef6i3"
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-qwen38-flash-next-v1 \
  vllm serve thousandml/Qwen3.8-Flash-Next-CQ-4bit \
  --quantization cq \
  --max-model-len 262144 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

For WSL2, add:

```bash id="uoqh38"
-e VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

## About Coded Quantization (CQ)

Learn more at **https://thousandml.com**.

**Coded Quantization (CQ)** is a calibration-free quantization approach for efficient LLM inference.

Traditional post-training quantization methods can depend on a calibration dataset to estimate quantization parameters. The resulting quantization can therefore be influenced by the distribution and composition of that calibration data.

CQ removes that calibration-data requirement. The goal is to retain strong fidelity to the original model while allowing the quantized representation to generalize beyond any particular calibration corpus.

## Base model

This checkpoint is derived from **Qwen/Qwen3.8-Flash-Next**.

Please refer to the original Qwen model card for architecture details, capabilities, usage guidance, limitations, and upstream licensing information.

## License

The base model is released under the applicable upstream Qwen license terms. See the repository license and the original Qwen model card for details.