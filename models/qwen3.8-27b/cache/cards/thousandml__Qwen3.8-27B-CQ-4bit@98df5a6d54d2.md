---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- qwen
- qwen3.8
- cq
- cq_quant
- quantization
- 4bit
- vllm
- text-generation
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-CQ-4bit

> **27B-parameter base model · Coded Quantization (CQ) 4-bit**
>
> **Note on Hugging Face's model-size display:** Hugging Face may report roughly **3B stored parameters** for this repository because CQ stores the weights in a packed 4-bit representation. The underlying model is **Qwen3.8-27B (27B parameters)**. The smaller number reflects the packed storage tensors counted by the Hub, not the parameter count of the original model architecture.

**Qwen3.8 27B quantized to 4-bit with Coded Quantization (CQ).**

**ThousandML:** https://thousandml.com

CQ is a **calibration-free 4-bit quantization method** designed to preserve the behavior of the original model without requiring a calibration dataset. Because CQ does not optimize the quantization around a fixed calibration corpus, it is designed to generalize beyond calibration-specific data and workloads.

This repository contains the **27B-parameter Qwen3.8 model** quantized with **Coded Quantization (CQ) 4-bit**.

## Quantization fidelity

### **CQ 4-bit · Qwen 3.8 27B · WikiText Raw**

| Metric | Result |
|---|---:|
| **Mean KL divergence** | **0.0282** |
| **Top-1 agreement** | **92.4%** |

Lower mean KL divergence indicates closer agreement between the quantized model's output distribution and the original model. Top-1 agreement measures how often the quantized model and original model select the same highest-probability next token.

## Why CQ?

**Calibration-free.** CQ does not require a representative calibration dataset before quantization.

This avoids tying the quantization procedure to a particular calibration corpus and is intended to provide better generalization beyond the data that would otherwise have been used for calibration.

**4-bit inference.** Model weights are quantized to 4-bit for substantially lower weight memory requirements than BF16/FP16 deployment.

**Native vLLM integration.** CQ models run through the CQ vLLM quantization plugin and custom CUDA kernels included in the Thousandml CQ Docker image.

## Model

| | |
|---|---|
| **Base model** | `Qwen/Qwen3.8-27B` |
| **Original model parameters** | **27B** |
| **HF displayed stored tensor count** | May appear as ~3B due to CQ 4-bit packing |
| **Quantization** | **Coded Quantization (CQ) 4-bit** |
| **Calibration** | **None — calibration-free** |
| **Runtime** | vLLM + CQ custom kernels |
| **Quantization name** | `cq_quant` |

## Run with Docker

The recommended way to run this model is with the CQ Docker image, which contains the compatible vLLM installation, CQ quantization plugin, and CQ CUDA kernels.

### 1. Make sure NVIDIA Docker support works

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.0.2-base-ubuntu22.04 \
  nvidia-smi
```

Your GPU should appear in the output.

### 2. Start Qwen3.8-27B-CQ-4bit

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-main \
  vllm serve thousandml/Qwen3.8-27B-CQ-4bit \
  --quantization cq
```

The model is then available through vLLM's OpenAI-compatible API on port `8000`.

### WSL2

If you are running Docker through WSL2 and vLLM reports that UVA is unavailable, enable vLLM's WSL2 pinned-memory support:

```bash
docker run --rm --gpus all \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-main \
  vllm serve thousandml/Qwen3.8-27B-CQ-4bit \
  --quantization cq
```

## Send a request

Once the server is ready:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "thousandml/Qwen3.8-27B-CQ-4bit",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantization in one paragraph."
      }
    ]
  }'
```

## Hugging Face authentication

For higher Hugging Face Hub rate limits, pass your Hugging Face token into the container:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  docker.io/thousandml/cq:4bit-main \
  vllm serve thousandml/Qwen3.8-27B-CQ-4bit \
  --quantization cq
```

On WSL2, combine it with:

```bash
-e VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

## About Coded Quantization (CQ)

Learn more at **https://thousandml.com**.

**Coded Quantization (CQ)** is a calibration-free quantization approach for efficient LLM inference.

Traditional post-training quantization methods can depend on a calibration dataset to estimate quantization parameters. The resulting quantization can therefore be influenced by the distribution and composition of that calibration data.

CQ removes that calibration-data requirement. The goal is to retain strong fidelity to the original model while allowing the quantized representation to generalize beyond any particular calibration corpus.

For this Qwen3.8 27B checkpoint, the CQ 4-bit model achieves:

- **0.0282 mean KL divergence** on WikiText Raw
- **92.4% top-1 agreement** with the original model on WikiText Raw
- **No calibration dataset required**

## Base model

This checkpoint is derived from **Qwen/Qwen3.8-27B**, a 27B-parameter Qwen3.8 model.

Please refer to the original Qwen model card for architecture details, capabilities, usage guidance, limitations, and upstream licensing information.

## License

The base model is released under the **Apache 2.0 License**. See the repository license and the original Qwen model card for applicable terms.
