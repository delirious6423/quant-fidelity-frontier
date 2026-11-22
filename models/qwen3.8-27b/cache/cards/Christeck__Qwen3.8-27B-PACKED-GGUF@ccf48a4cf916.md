---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - exl3
  - trellis
  - apple-silicon
  - metal
  - quantized
language:
  - en
  - es
---

# Qwen3.8-27B — Packed GGUF (EXL3 trellis) for Metal Trellis

A lossless repack of Turboderp's EXL3 trellis codes for **Qwen3.8-27B** into the Packed GGUF format used by the [Metal Trellis](https://github.com/Christeck-AI/metal-trellis) engine (Apple Silicon / Metal). No retraining, no fine-tuning: the trellis codes are identical to the EXL3 source; only the container changed.

**The engine is the product.** This file is the courtesy copy, published so nobody has to repeat the conversion work (which originally ran on rented Blackwell/A100 hardware). If you want to build your own Packed GGUFs from any EXL3 checkpoint, the instructions live in the Metal Trellis repo.

## How to run

This is **not** a stock llama.cpp GGUF. It requires the Metal Trellis fork, which knows how to decode the trellis-packed tensors on the fly with Metal kernels:

- Engine: https://github.com/Christeck-AI/metal-trellis
- Platform: Apple Silicon (Metal), macOS

## Honest measurements (M1 Max, 64 GB)

- Decode: **2.70 tok/s** — the dense 27B is a research artifact, not a daily driver.
- Correctness: **100% token-identical** vs the Q8_0 reference on the verification battery.
- Quality: EXL3 at 3.00 bpw measured indistinguishable from BF16 in NLL (confidence interval includes 0).
- The win here is **memory** (10.84 GiB for a 27B dense model), not speed. For practical use, see the MoE sibling: [Christeck/Qwen3.5-35B-A3B-PACKED-GGUF](https://huggingface.co/Christeck/Qwen3.5-35B-A3B-PACKED-GGUF).

## Attribution and changes

- Base model: **Qwen3.8-27B** by Alibaba Cloud (Qwen team), Apache License 2.0.
- Quantization: **EXL3 trellis quantization by Turboderp** (ExLlamaV3, MIT), distributed under the base model's license.
- Changes made by Christeck Labs (Apache 2.0 §4b declaration): lossless repack of the EXL3 trellis codes into the Metal Trellis Packed GGUF container format. No weights were retrained or altered.

This model is distributed under the Apache License 2.0, "AS IS", without warranties of any kind.

## Resumen (español)

Reempaque sin pérdida de los códigos trellis EXL3 de Turboderp para Qwen3.8-27B, en el formato Packed GGUF del motor [Metal Trellis](https://github.com/Christeck-AI/metal-trellis) para Apple Silicon. Sin reentrenamiento: los códigos son idénticos a la fuente EXL3; solo cambió el contenedor. **El producto es el motor y sus instrucciones**; este archivo es la copia de cortesía para que nadie repita el trabajo de conversión. No es un GGUF estándar de llama.cpp: requiere el fork Metal Trellis. Medido en M1 Max 64 GB: 2.70 tok/s, 100% token-idéntico frente a Q8_0 — el denso es investigación; la ganancia es memoria. Licencia Apache 2.0 heredada del modelo base (Alibaba/Qwen), con la cuantización EXL3 acreditada a Turboderp.
