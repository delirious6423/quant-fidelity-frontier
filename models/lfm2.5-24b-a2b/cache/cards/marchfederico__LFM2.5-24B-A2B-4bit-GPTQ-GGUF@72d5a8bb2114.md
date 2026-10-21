---
base_model: LiquidAI/LFM2.5-24B-A2B
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - gptq
  - quantized
  - lfm2
  - moe
---

# LFM2.5-24B-A2B 4-bit GPTQ-K (GGUF, ~5.0 bpw)

A 4-bit-class GGUF of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) that runs in stock
llama.cpp (`lfm2moe`, standard K-quant and Q8_0 kernels, no fork needed).

**Modified work notice (LFM Open License v1.0, section 4(b)):** the weights in this file are a quantized modification of
LiquidAI/LFM2.5-24B-A2B. The original license is included as `LICENSE` and applies to this file, including the
commercial-use limitation in its section 5.

## Credits

- **Dom Federico** ([GitHub dfed25](https://github.com/dfed25), [Hugging Face dfed24](https://huggingface.co/dfed24)): the
  quantization recipe this file is built on, from [mlx-gptq](https://github.com/dfed25/mlx-gptq): sequential GPTQ with
  error feedback, MSE range search, Lloyd grid fit and the Hessian-metric refinement and grid refit.
- **unsloth**: the per-tensor type allocation, taken from their UD-Q4_K_XL file for this model.
- **Liquid AI**: the base model, [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

## How it was made

- **Quantizer:** sequential GPTQ with error feedback, MSE range search, Lloyd grid fit and Hessian-metric refinement / grid
  refit, fitted directly under the llama.cpp K-quant constraints (super-blocks of 256 with 6-bit sub-scales), so the codes
  pack into Q4_K / Q5_K / Q6_K exactly. Dom Federico's recipe
  ([mlx-gptq](https://github.com/dfed25/mlx-gptq)), ported to PyTorch with mixture-of-experts support.
- **Experts:** each of the 64 experts per layer gets its own Hessian from the tokens routed to it (gate and up share it),
  shrunk toward the layer's pooled expert statistics for rarely routed experts. All-zero rows present in the original
  checkpoint are kept as exact zeros.
- **Per-tensor allocation:** the tensor types follow unsloth's UD-Q4_K_XL file for this model (Q4_K expert gate/up, Q5_K
  expert down, Q5_K/Q6_K in layers 7-8, Q8_0 attention, dense layers and embedding, Q4_K short-conv projections), so
  the file is byte-for-byte the same size as theirs. Credit to unsloth for the allocation; their calibration data and
  imatrix were not used.
- **Calibration:** 1,024 windows of 512 tokens from a mixed English / code / multilingual text.

## Fidelity

KL divergence from the bf16 model and top-1 agreement, 6 windows of 2,048 tokens per held-out text, all files measured
with the same harness (lower KL, higher top-1 is better):

| file | size | wiki | code | multilingual |
|---|---|---|---|---|
| **this file** | 14,188 MiB | **0.104 / 84.9%** | **0.086 / 92.1%** | **0.122 / 85.9%** |
| unsloth UD-Q4_K_XL | 14,188 MiB | 0.105 / 85.1% | 0.089 / 91.7% | 0.126 / 85.6% |
| LiquidAI Q4_K_M | 13,853 MiB | 0.209 / 78.8% | 0.184 / 87.7% | 0.353 / 76.5% |

No task benchmarks were run; the KL suite is a fidelity ranking, not a capability score. This MoE is more sensitive to
quantization than dense models of similar size (top-4 sigmoid routing over 64 experts), so all 4-bit files sit well above
the KL a dense model would show.

## Usage

```bash
llama-cli -m LFM2.5-24B-A2B-GPTQ-Q4_K.gguf -p "The capital of France is" -n 32
```
