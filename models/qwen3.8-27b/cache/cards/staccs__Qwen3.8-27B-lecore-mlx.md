---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
tags: [mlx, quantized, mixed-precision, lecore, qwen3_5]
---
# Qwen3.8-27B-lecore-mlx

MLX mixed-precision quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with the bit width of every
linear layer picked by **leCore Unicron `requantize`** (per-tensor bits chosen by measurement), ported to GPU for this model size.

## How the bits were chosen
1. Each decoder linear tensor was quantized **alone** (MLX affine, group 64 during measurement) at 2, 3 and 4 bits; the damage was measured as
   KL(fp16 || quantized) on next-token distributions over a calibration mix (3066 WikiText-2 *train* tokens,
   1022 Python-source tokens, and leCore's structured probe: prose, code, SQL, markdown).
2. Bits were assigned greedily: start at 4 bits everywhere, repeatedly take the downgrade with the least measured damage per byte
   saved until the predicted file size fit the budget. Embeddings 4-bit, lm_head 6-bit, group size 64.
3. Converted with `mlx_lm.convert` and a per-layer `quant_predicate`. Bit histogram (linear tensors): {'2': 0, '3': 0, '4': 496, '8': 0, '16': 0}
   **Note:** at this size budget the allocator kept *every* linear tensor at the top rung (4 bits), so the per-tensor selection made no downgrades. In practice this export is **uniform 4-bit (group 64)**, with 4-bit embeddings and a 6-bit lm_head. It is not a mixed-precision mix, and the fair baseline is stock MLX 4-bit, not 3-bit
   **Note:** at this size budget the allocator kept *every* linear tensor at the top rung (4 bits), so the per-tensor selection made no downgrades. In practice this export is **uniform 4-bit (group 64)**, with 4-bit embeddings and a 6-bit lm_head. It is not a mixed-precision mix, and the fair baseline is stock MLX 4-bit, not 3-bit.

## Measured results (held-out: WikiText-2 *test* + a different Python-source slice, seq 512)
Perplexity was computed on the exact shipped weights (MLX tensors dequantized with `mx.dequantize`, run in bf16 on GPU).
"bpw" = integer bits + 0.5 for the bf16 scale/bias per group of 64.

| Model | Size (GB, weights) | bpw | WikiText-2 ppl (32704 tok) | Python-code ppl (8176 tok) |
|---|---|---|---|---|
| bf16 reference | – | 16 | 7.8880 | 2.5651 |
| **this repo (leCore Unicron mix)** | 15.45 | 4.595 | 8.1249 | 2.7038 |
| stock MLX 3-bit, group 64 | 11.77 | 3.500 | 9.2464 | 3.5006 |

## Benchmarks (lm-eval-harness 0.4.9, run on the shipped weights dequantized to bf16 on GPU)
MMLU 5-shot on the first 10 questions of each of the 57 subjects; GSM8K 5-shot on the first 100 test problems (greedy, 320 new tokens, no chat template). These are **subsets**, so expect about ±2–4 points of sampling noise.

| Model | MMLU (5-shot) | GSM8K strict | GSM8K flexible |
|---|---|---|---|
| **leCore mix (this repo)** | 84.9% | 91.0% | 89.0% |
| stock 3-bit affine, group 64 (simulated) | 81.4% | 60.0% | 62.0% |

Only measured numbers are reported. Downstream benchmarks are limited to the MMLU/GSM8K subsets above. Short-context perplexity does not tell you everything about quality, so check outputs for your own use case.

## Usage (Apple Silicon)
```bash
pip install -U mlx-lm
mlx_lm.generate --model staccs/Qwen3.8-27B-lecore-mlx --prompt "Hello" --max-tokens 256 --max-kv-size 8192
```
Weights are ~15.45 GB. On a 36 GB Mac you may need to raise the GPU wired-memory limit, e.g. `sudo sysctl iogpu.wired_limit_mb=30000`.

## License
Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), released by the Qwen team under the **Apache License 2.0**. Only quantization was done here, with no retraining. **Text-only:** the vision tower and the MTP head are not included in this export. The hybrid Gated-DeltaNet layers are quantized per tensor like every other linear layer.
