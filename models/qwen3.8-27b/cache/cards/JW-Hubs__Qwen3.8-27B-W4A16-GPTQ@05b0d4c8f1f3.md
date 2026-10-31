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
vllm serve  J-Fraudster/Qwen3.8-27B-W4A16-GPTQ \
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

## 📚 Acknowledgments

* Quantization performed using [Intel AutoRound](https://github.com/intel/auto-round).
* Base architecture provided by the [Qwen Team](https://github.com/QwenLM/Qwen).
