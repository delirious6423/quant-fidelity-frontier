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

| Benchmark | Problems × draws | Output cap | BF16<br>55.6 GB | QDA W4A4<br>21.9 GB | NVIDIA NVFP4<br>21.9 GB | RedHatAI NVFP4<br>24.7 GB | unsloth NVFP4<br>23.4 GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIME 2025 | 30 × 32 | 81,920 | 88.4 ± 4.1 | 87.1 ± 3.7 | **91.1 ± 3.1** | 87.1 ± 4.4 | 88.9 ± 3.5 |
| HMMT Feb 2026 | 33 × 32 | 81,920 | 72.6 ± 5.5 | 73.4 ± 5.5 | 74.9 ± 5.0 | **76.8 ± 5.0** | 72.5 ± 5.6 |
| GPQA-Diamond | 198 × 16 | 32,768 | 82.4 ± 2.1 | 80.8 ± 2.2 | 81.1 ± 2.2 | 80.2 ± 2.3 | **81.7 ± 2.2** |
| IFBench | 300 × 16 | 32,768 | 67.0 ± 2.1 | 68.0 ± 2.1 | 68.6 ± 2.1 | 67.6 ± 2.1 | **68.7 ± 2.1** |
| MMLU-Pro | 12,032 × 1 | 32,768 | 83.6 ± 0.3 | 83.1 ± 0.3 | **83.4 ± 0.3** | 83.2 ± 0.3 | 83.2 ± 0.3 |
| LiveCodeBench | 454 × 4 | 81,920 | 86.7 ± 1.4 | 85.2 ± 1.4 | 85.8 ± 1.4 | **87.3 ± 1.4** | 85.5 ± 1.4 |
| **Average** | | | 80.1 ± 1.3 | 79.6 ± 1.2 | **80.8 ± 1.1** | 80.4 ± 1.2 | 80.1 ± 1.2 |

Thinking mode, pass@1 averaged over the draws in the table. Each cell is score ± 1 SE, all in percent; the best NVFP4 score in a row is bold. SE = sample standard deviation (n - 1) of the per-problem mean across draws, divided by sqrt(number of problems); the average row's SE is sqrt(sum of the six SEs squared) / 6, the benchmarks being independent problem sets. Sizes are the sum of the `*.safetensors` files in decimal GB.

Source of every figure above: our own runs. All five checkpoints — the BF16 base model, this one, and the NVFP4 checkpoints published by NVIDIA, RedHatAI and unsloth — were served by vLLM 0.33.0 on one GB300 with the same problems, draws, seeds, sampling and output-token caps. MMLU-Pro was run greedily with `reasoning_effort: medium`.

<details><summary>Truncation (share of generations stopped at the output cap)</summary>

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

</details>

On this six-benchmark average our checkpoint is the lowest of the four NVFP4 checkpoints, 1.2 points below
NVIDIA's at the same size. The gap sits in the two math sets (63 problems), so BF16, QDA and NVIDIA were also run
on three further MathArena sets, asked and graded the same way:

| | AIME 2025 | HMMT Feb 2026 | AIME 2026 | HMMT Nov 2025 | HMMT Feb 2025 | all 153 problems |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 88.4 | 72.6 | 87.9 | 79.2 | 78.0 | 81.1 |
| QDA W4A4 | 87.1 | 73.4 | 89.2 | 79.9 | 82.7 | 82.3 |
| NVIDIA | 91.1 | 74.9 | 89.3 | 77.9 | 80.9 | 82.7 |

Source: our own runs, same harness, server and settings as the table above. Over all 153 problems NVIDIA − QDA
is +0.4 points, with a paired 95 % interval of −1.2 to +2.1: on math the two checkpoints are not
distinguishable. A second set of sampling seeds moved one checkpoint's score on a 30-problem set by up to 2.7
points.

## Evaluation settings

The settings the numbers above were run with:

| Benchmark | Problems × draws | Output cap | Sampling | Thinking / chat template | Engine window |
|---|---:|---:|---|---|---:|
| AIME 2025 | 30 × 32 | 81,920 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` | 262,144 |
| HMMT Feb 2026 | 33 × 32 | 81,920 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` | 262,144 |
| GPQA-Diamond | 198 × 16 | 32,768 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` | 262,144 |
| IFBench | 300 × 16 | 32,768 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` | 262,144 |
| MMLU-Pro | 12,032 × 1 | 32,768 | greedy (temperature 0) | thinking on, `</think>`, `reasoning_effort: medium` | 262,144 |
| LiveCodeBench | 454 × 4 | 81,920 | temperature 1.0, top-p 0.95, top-k 20 | thinking on, `</think>`, `reasoning_effort: medium` | 262,144 |

The server read the sampling from this checkpoint's `generation_config.json` (temperature 1.0, top-p 0.95, top-k 20, `do_sample: true`) and was given `reasoning_effort: medium` in the chat template; MMLU-Pro was run greedily at the same top-p and top-k. `</think>` is the end-of-thinking marker the harness splits each generation on. The engine window is the served context length.

`qda eval` with the `full` suite (`suites: [{full: {}}]` in its config) runs the same six benchmarks with the same draws, caps and chat-template argument as the table above, with two differences: it samples MMLU-Pro instead of running it greedily, and it serves the model with a 114,688-token context window rather than 262,144 (its IFEval cap is 32,768; IFEval is not one of the six benchmarks here). The numbers above are the ones this checkpoint was measured with.

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

**Calibration data.** 162 answers that the BF16 model wrote to 162 prompts (40 science, 40 math, 40 code, 30
instruction-following, 12 chat), cut into 52 samples of 4,096 tokens. Checked against the MMLU-Pro test set
(12,032 questions) by 20-word windows, 13-word windows and exact containment of each question: no overlap.

## Reproduce

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W4A4/*" --local-dir inputs
cd inputs/QDA-Qwen3.8-27B-W4A4
sha256sum -c SHA256SUMS
qda build qwen3.8-27b-w4a4-nvfp4.yaml --set out=QDA-Qwen3.8-27B-W4A4-rebuild --set "devices=[0]"
```

Needs QDA with NVFP4 support, commit `10f56f50` of [Radioheading/qda](https://github.com/Radioheading/qda/commit/10f56f50c484a45beeca4fc855ee3edb0b5d89fc) (the NVFP4 pull request, not yet in the main branch; this card will name the main-branch commit once it is merged), and one Blackwell GPU. `devices=[0]` keeps the build on one card: on more cards the calibration statistics are summed in another order and the bytes differ. The folder holds the recipe, the
calibration file and the prompts it answered, `SHA256SUMS` over them and `CHECKPOINT.sha256` over this
checkpoint. A rebuild on one GB300 takes about 23 min and is byte-identical to this checkpoint's weights.

## License

Apache-2.0, the same as the base model.
