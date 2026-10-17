---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- quantization
- auto-round
- gptq
- vlm
- 4bit
- text-generation
- image-text-to-text
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B (W4A16 Quantized via AutoRound)

This repository contains a **W4A16 (4-bit weights, 16-bit activations)** quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) generated using Intel's [AutoRound](https://github.com/intel/auto-round) algorithm.

---

## ⚡ Quantization Details

The model was calibrated and quantized using production-grade settings to minimize accuracy degradation while significantly lowering VRAM requirements:

* **Quantization Algorithm:** [AutoRound](https://github.com/intel/auto-round)
* **Bits / Precision:** W4A16 (4-bit integer weights, 16-bit activation)
* **Group Size:** 64 (provides higher reconstruction fidelity than standard 128)
* **Symmetric (`sym`):** `True`
* **Calibration Samples (`nsamples`):** 512
* **Tuning Iterations (`iters`):** 600
* **Vision Tower (`quant_nontext_module`):** `False` (Kept in **BF16** to preserve visual reasoning and OCR precision)
* **Special Modules (`layer_config`):** Multi-Token Prediction (`mtp`, `mtp.fc`) layers preserved in native `bfloat16`.

---

## 🚀 Usage & Quickstart

### 1. Inference via vLLM

For high-throughput production serving:

```bash
vllm serve Vishva007/Qwen3.8-27B-W4A16-AutoRound \
    --quantization auto-round \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90

```

*(Note: For the GPTQ format repo, you can set `--quantization gptq` if required by your backend).*

---

## 📊 VRAM & Performance Benefits

* **Original Model (BF16):** ~54 GB VRAM required (needs 2x A100/A6000 or 80GB VRAM)
* **Quantized Model (W4A16 Group 64):** ~16–18 GB VRAM (can easily run on a single **RTX 3090 / 4090 / A5000 24GB**)
* **Speedup:** Reduced memory bandwidth bottleneck leading to faster decoding token speeds.

---

### 🚀 Deploy on RunPod

One-click launch environments pre-configured with PyTorch, CUDA, and dependencies for fine-tuning or quantization.

> 🎁 **Need GPU compute?** Sign up via [RunPod](https://runpod.io?ref=iabrlp7z) and get **$5–$500 in free credits** when you add your first $10.

#### PyTorch 2.14

| Template | CUDA Version | Docker Image | Template ID | Deploy |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch 2.14 (CUDA 12.6)** | 12.6 | `vishva123/cuda-12.6-pytorch-2.14-runpod` | `d7lxsa4w9m` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=d7lxsa4w9m&ref=iabrlp7z) |
| **PyTorch 2.14 (CUDA 13.0)** | 13.0 | `vishva123/cuda-13.0-pytorch-2.14-runpod` | `yk0y6j6rpg` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=yk0y6j6rpg&ref=iabrlp7z) |
| **PyTorch 2.14 (CUDA 13.2)** | 13.2 | `vishva123/cuda-13.2-pytorch-2.14-runpod` | `gsp4gwx0nw` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=gsp4gwx0nw&ref=iabrlp7z) |

#### PyTorch 2.13

| Template | CUDA Version | Docker Image | Template ID | Deploy |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch 2.13 (CUDA 12.6)** | 12.6 | `vishva123/cuda-12.6-pytorch-2.13-runpod` | `gmlupxnxfk` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=gmlupxnxfk&ref=iabrlp7z) |
| **PyTorch 2.13 (CUDA 13.0)** | 13.0 | `vishva123/cuda-13.0-pytorch-2.13-runpod` | `y3j8xvk4f4` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=y3j8xvk4f4&ref=iabrlp7z) |
| **PyTorch 2.13 (CUDA 13.2)** | 13.2 | `vishva123/cuda-13.2-pytorch-2.13-runpod` | `vigpissn5w` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=vigpissn5w&ref=iabrlp7z) |

#### PyTorch 2.12

| Template | CUDA Version | Docker Image | Template ID | Deploy |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch 2.12 (CUDA 12.6)** | 12.6 | `vishva123/cuda-12.6-pytorch-2.12-runpod` | `ctmz86zmf0` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=ctmz86zmf0&ref=iabrlp7z) |
| **PyTorch 2.12 (CUDA 13.0)** | 13.0 | `vishva123/cuda-13.0-pytorch-2.12-runpod` | `qjko5yiwzi` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=qjko5yiwzi&ref=iabrlp7z) |
| **PyTorch 2.12 (CUDA 13.2)** | 13.2 | `vishva123/cuda-13.2-pytorch-2.12-runpod` | `ifg6xmye0f` | [![Deploy to RunPod](https://img.shields.io/badge/RunPod-Deploy%20Pod-purple?logo=runpod&logoColor=white)](https://runpod.io/console/deploy?template=ifg6xmye0f&ref=iabrlp7z) |

---

## 📚 Acknowledgments

* Quantization performed using [Intel AutoRound](https://github.com/intel/auto-round).
* Base architecture provided by the [Qwen Team](https://github.com/QwenLM/Qwen).
