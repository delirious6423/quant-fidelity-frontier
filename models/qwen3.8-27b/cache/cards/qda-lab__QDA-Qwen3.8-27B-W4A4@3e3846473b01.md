---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [qwen, qwen3.8, qda, quantized, nvfp4, fp4, w4a4, fp8, compressed-tensors, vllm, blackwell]
---

# QDA-Qwen3.8-27B-W4A4

NVFP4 quantization (4-bit weights and 4-bit activations, with some layers in FP8) of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for NVIDIA Blackwell GPUs.
**2.5× smaller than BF16 (21.9 GB vs. 55.6 GB), keeps 99.4 % of BF16 accuracy.**

## Evaluation

| Benchmark | Problems × draws | Output cap | BF16 (55.6 GB) | QDA W4A4 (21.9 GB) | NVIDIA NVFP4 (21.9 GB) | RedHatAI NVFP4 (24.7 GB) | unsloth NVFP4 (23.4 GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIME 2025 | 30 × 32 | 81,920 | 88.4 ± 4.1 | 87.1 ± 3.7 | **91.1 ± 3.1** | 87.1 ± 4.4 | 88.9 ± 3.5 |
| HMMT Feb 2026 | 33 × 32 | 81,920 | 72.6 ± 5.5 | 73.4 ± 5.5 | 74.9 ± 5.0 | **76.8 ± 5.0** | 72.5 ± 5.6 |
| GPQA-Diamond | 198 × 16 | 32,768 | 82.4 ± 2.1 | 80.8 ± 2.2 | 81.1 ± 2.2 | 80.2 ± 2.3 | **81.7 ± 2.2** |
| IFBench | 300 × 16 | 32,768 | 67.0 ± 2.1 | 68.0 ± 2.1 | 68.6 ± 2.1 | 67.6 ± 2.1 | **68.7 ± 2.1** |
| MMLU-Pro | 12,032 × 1 | 32,768 | 83.6 ± 0.3 | 83.1 ± 0.3 | **83.4 ± 0.3** | 83.2 ± 0.3 | 83.2 ± 0.3 |
| LiveCodeBench | 454 × 4 | 81,920 | 86.7 ± 1.4 | 85.2 ± 1.4 | 85.8 ± 1.4 | **87.3 ± 1.4** | 85.5 ± 1.4 |
| **Average** | | | 80.1 ± 1.3 | 79.6 ± 1.2 | **80.8 ± 1.1** | 80.4 ± 1.2 | 80.1 ± 1.2 |

Thinking mode, pass@1 averaged over the draws in the table. Each cell is score ± 1 SE, all in percent. SE = sample standard deviation (n - 1) of the per-problem mean across draws, divided by sqrt(number of problems); the average row's SE is sqrt(sum of the six SEs squared) / 6, the benchmarks being independent problem sets. Sizes are the sum of the `*.safetensors` files in decimal GB.

Source: our own runs. All five checkpoints — the BF16 base model, this one, and the NVFP4 checkpoints published by NVIDIA, RedHatAI and unsloth — were run with the same problems, draws, seeds, sampling and output-token caps. MMLU-Pro was run greedily with `reasoning_effort: medium`.

### Paired differences

QDA W4A4 minus the checkpoint named in each column, in percentage points: difference [95 % interval]. The difference is the mean over problems of the per-problem difference between the two checkpoints' draw means; the interval is the difference ± 1.96 × the paired SE, the sample standard deviation (n - 1) of those per-problem differences divided by sqrt(number of problems). The average row's difference is the mean of the six differences and its SE is sqrt(sum of the six paired SEs squared) / 6.

| Benchmark | QDA W4A4 − BF16 | QDA W4A4 − NVIDIA | QDA W4A4 − RedHatAI | QDA W4A4 − unsloth |
|---|---:|---:|---:|---:|
| AIME 2025 | −1.35 [−3.55, +0.84] | −4.06 [−8.04, −0.09] | +0.00 [−4.20, +4.20] | −1.77 [−4.75, +1.21] |
| HMMT Feb 2026 | +0.76 [−2.88, +4.39] | −1.52 [−5.81, +2.78] | −3.41 [−7.72, +0.90] | +0.85 [−3.36, +5.06] |
| GPQA-Diamond | −1.52 [−2.99, −0.04] | −0.28 [−1.81, +1.25] | +0.60 [−0.90, +2.10] | −0.82 [−2.16, +0.52] |
| IFBench | +1.08 [−0.31, +2.47] | −0.58 [−1.81, +0.64] | +0.46 [−0.90, +1.82] | −0.69 [−1.96, +0.59] |
| MMLU-Pro | −0.46 [−0.91, +0.00] | −0.30 [−0.77, +0.17] | −0.07 [−0.53, +0.40] | −0.07 [−0.53, +0.38] |
| LiveCodeBench | −1.49 [−2.85, −0.12] | −0.61 [−2.14, +0.93] | −2.09 [−3.47, −0.71] | −0.28 [−1.66, +1.11] |
| **Average** | −0.50 [−1.32, +0.32] | −1.23 [−2.29, −0.16] | −0.75 [−1.84, +0.33] | −0.46 [−1.41, +0.48] |

### Truncation

Share of generations stopped at the output cap.

| benchmark | BF16 | QDA W4A4 | NVIDIA | RedHatAI | unsloth |
|---|---:|---:|---:|---:|---:|
| AIME 2025 | 1.1 % | 3.2 % | 3.0 % | 0.5 % | 1.6 % |
| HMMT Feb 2026 | 0.1 % | 1.8 % | 1.2 % | 0.8 % | 0.1 % |
| GPQA-Diamond | 3.0 % | 3.4 % | 2.3 % | 2.9 % | 3.0 % |
| IFBench | 0.7 % | 0.9 % | 0.9 % | 0.8 % | 1.1 % |
| MMLU-Pro | <0.1 % | 0 % | 0 % | <0.1 % | 0 % |
| LiveCodeBench | 6.7 % | 7.2 % | 6.4 % | 6.3 % | 7.5 % |
| Average | 2.0 % | 2.8 % | 2.3 % | 1.9 % | 2.2 % |

Truncation = generations whose recorded finish reason is 'length' over all generations of the cell (problems × draws). The average row is the unweighted mean of the six rows.

### Three further math sets

BF16, QDA W4A4 and NVIDIA were also run on three further MathArena sets, asked and graded the same way as the first two:

| | AIME 2025 | HMMT Feb 2026 | AIME 2026 | HMMT Nov 2025 | HMMT Feb 2025 | all 153 problems |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 88.4 | 72.6 | 87.9 | 79.2 | 78.0 | 81.1 |
| QDA W4A4 | 87.1 | 73.4 | 89.2 | 79.9 | 82.7 | 82.3 |
| NVIDIA | 91.1 | 74.9 | 89.3 | 77.9 | 80.9 | 82.7 |
| QDA W4A4 − BF16 | −1.35 [−3.55, +0.84] | +0.76 [−2.88, +4.39] | +1.25 [−1.45, +3.95] | +0.73 [−2.07, +3.53] | +4.69 [−0.30, +9.68] | +1.21 [−0.33, +2.75] |
| QDA W4A4 − NVIDIA | −4.06 [−8.04, −0.09] | −1.52 [−5.81, +2.78] | −0.10 [−2.11, +1.90] | +1.98 [−0.67, +4.63] | +1.77 [−2.28, +5.83] | −0.41 [−2.02, +1.20] |

Source: our own runs, same harness and settings as the table above. The last column pools all 153 problems. The two difference rows are paired over problems (same definition as above, with the interval over the 30 to 33 problems of a set, or over all 153).

### Evaluation settings

| Benchmark | Problems × draws | Output cap | Sampling | Thinking / chat template |
|---|---:|---:|---|---|
| AIME 2025 | 30 × 32 | 81,920 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` |
| HMMT Feb 2026 | 33 × 32 | 81,920 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` |
| GPQA-Diamond | 198 × 16 | 32,768 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` |
| IFBench | 300 × 16 | 32,768 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` |
| MMLU-Pro | 12,032 × 1 | 32,768 | greedy (temperature 0) | thinking on, `</think>`, `reasoning_effort: medium` |
| LiveCodeBench | 454 × 4 | 81,920 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` |

`</think>` is the end-of-thinking marker the harness splits each generation on.

## Usage

Runs on NVIDIA Blackwell data-center GPUs (B200, B300, GB200, GB300); evaluated with vLLM 0.33.0 on one GB300.

```bash
vllm serve qda-lab/QDA-Qwen3.8-27B-W4A4
```

## How it was made

The MLPs are NVFP4: E2M1 weights with one E4M3 scale per 16 inputs and an FP32 tensor scale, activations
quantized the same way at run time. FP8 W8A8 (per-channel weights, dynamic per-token activations) holds the
self-attention projections of the 16 full-attention layers, every linear-attention `out_proj`, the
linear-attention `in_proj_qkv` / `in_proj_z` of layers 0–41, and `lm_head`; layers 42 and up keep NVFP4
`in_proj` weights. The Gated DeltaNet `in_proj_a` / `in_proj_b`, the embeddings and the vision tower stay in
BF16. Weights were quantized with GPTQ; the recipe was chosen by QDA's automated search under a size budget
equal to NVIDIA's checkpoint.

## Reproduce

Install QDA 0.1.0 (Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/Radioheading/qda.git && cd qda
uv venv -p 3.12 --managed-python && . .venv/bin/activate
uv pip install -c constraints.txt "./packages/qda-quant[hf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
```

Then:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W4A4/*" --local-dir inputs
cd inputs/QDA-Qwen3.8-27B-W4A4
sha256sum -c SHA256SUMS
qda build qwen3.8-27b-w4a4-nvfp4.yaml --set out=QDA-Qwen3.8-27B-W4A4-rebuild --set "devices=[0]"
```

Needs QDA 0.1.0 and one Blackwell GPU. Original model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the revision the recipe builds from. A rebuild's KL to BF16 is within ±5 % of this checkpoint's; the weights are not bit-identical. The folder holds the recipe, the calibration data, `SHA256SUMS` over them and `CHECKPOINT.sha256` over this checkpoint; its `README.md` describes the calibration data and the evaluation setup. A rebuild on one GB300 takes about 23 min.

## License

Apache-2.0, the same as the base model.

## Citation

TBD.
