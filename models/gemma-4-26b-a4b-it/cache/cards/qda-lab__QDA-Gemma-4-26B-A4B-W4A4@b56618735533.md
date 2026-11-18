---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [gemma, gemma4, moe, qda, quantized, nvfp4, fp4, w4a4, fp8, compressed-tensors, vllm, blackwell]
---

# QDA-Gemma-4-26B-A4B-W4A4

NVFP4 quantization (4-bit weights and 4-bit activations, with attention in FP8) of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) for NVIDIA Blackwell GPUs.
**3.1× smaller than BF16 (16.9 GB vs. 51.6 GB), keeps 97.6 % of BF16 accuracy.**

## Evaluation

| Benchmark | Problems × draws | Output cap | BF16 (51.6 GB) | QDA W4A4 (16.9 GB) | NVIDIA NVFP4 (18.8 GB) | RedHatAI NVFP4 (16.4 GB) | unsloth NVFP4 (16.9 GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIME 2025 | 30 × 32 | 32,768 | 85.5 ± 5.1 | 82.7 ± 5.8 | 84.1 ± 5.4 | 70.1 ± 6.8 | 82.1 ± 5.8 |
| HMMT Feb 2026 | 33 × 32 | 32,768 | 71.6 ± 6.3 | 67.2 ± 6.6 | 68.7 ± 6.6 | 54.4 ± 7.3 | 64.0 ± 6.7 |
| GPQA-Diamond | 198 × 16 | 32,768 | 80.6 ± 2.3 | 78.6 ± 2.4 | 79.2 ± 2.4 | 75.2 ± 2.4 | 77.9 ± 2.5 |
| IFBench | 300 × 16 | 32,768 | 80.6 ± 1.9 | 79.9 ± 1.9 | 80.9 ± 1.9 | 78.1 ± 1.9 | 79.3 ± 1.9 |
| MMLU-Pro | 12,032 × 1 | 32,768 | 84.1 ± 0.3 | 83.4 ± 0.3 | 83.4 ± 0.3 | 80.2 ± 0.4 | 83.2 ± 0.3 |
| LiveCodeBench | 454 × 4 | 32,768 | 81.3 ± 1.6 | 80.2 ± 1.7 | 79.5 ± 1.7 | 70.9 ± 1.9 | 77.5 ± 1.7 |
| **Average** | | | 80.6 ± 1.5 | 78.7 ± 1.6 | 79.3 ± 1.5 | 71.5 ± 1.8 | 77.3 ± 1.6 |

Thinking mode, pass@1 averaged over the draws in the table. Each cell is score ± 1 SE, all in percent. SE = sample standard deviation (n - 1) of the per-problem mean across draws, divided by sqrt(number of problems); the average row's SE is sqrt(sum of the six SEs squared) / 6, the benchmarks being independent problem sets. Sizes are the sum of the `*.safetensors` files in decimal GB.

Source: our own runs. All five checkpoints — the BF16 base model, this one, and the NVFP4 checkpoints published by NVIDIA, RedHatAI and unsloth — were run with the same problems, draws, seeds, sampling and output-token caps.


### Truncation

Share of generations stopped at the output cap.

| benchmark | BF16 | QDA W4A4 | NVIDIA | RedHatAI | unsloth |
|---|---:|---:|---:|---:|---:|
| AIME 2025 | 8.9 % | 9.4 % | 8.8 % | 20.8 % | 9.8 % |
| HMMT Feb 2026 | 21.2 % | 23.2 % | 20.5 % | 31.4 % | 25.1 % |
| GPQA-Diamond | 1.3 % | 2.0 % | 1.0 % | 0.8 % | 0.9 % |
| IFBench | 0.6 % | 1.0 % | 0.9 % | 1.6 % | 1.8 % |
| MMLU-Pro | 0.2 % | 0.3 % | 0.1 % | 0.3 % | 0.1 % |
| LiveCodeBench | 3.6 % | 5.3 % | 3.3 % | 12.4 % | 5.0 % |
| Average | 5.9 % | 6.9 % | 5.8 % | 11.2 % | 7.1 % |

Truncation = generations whose recorded finish reason is 'length' over all generations of the cell (problems × draws). The average row is the unweighted mean of the six rows.

### Evaluation settings

| Benchmark | Problems × draws | Output cap | Sampling | Thinking / chat template |
|---|---:|---:|---|---|
| AIME 2025 | 30 × 32 | 32,768 | temperature 1.0, top-p 0.95, top-k 64 | thinking on, `<channel\|>` |
| HMMT Feb 2026 | 33 × 32 | 32,768 | temperature 1.0, top-p 0.95, top-k 64 | thinking on, `<channel\|>` |
| GPQA-Diamond | 198 × 16 | 32,768 | temperature 1.0, top-p 0.95, top-k 64 | thinking on, `<channel\|>` |
| IFBench | 300 × 16 | 32,768 | temperature 1.0, top-p 0.95, top-k 64 | thinking on, `<channel\|>` |
| MMLU-Pro | 12,032 × 1 | 32,768 | temperature 1.0, top-p 0.95, top-k 64 | thinking on, `<channel\|>` |
| LiveCodeBench | 454 × 4 | 32,768 | temperature 1.0, top-p 0.95, top-k 64 | thinking on, `<channel\|>` |

The `<channel|>` marker is the end-of-thinking marker the harness splits each generation on.

## Usage

Runs on NVIDIA Blackwell data-center GPUs (B200, B300, GB200, GB300); evaluated with vLLM 0.33.0 on one GB300.

```bash
vllm serve qda-lab/QDA-Gemma-4-26B-A4B-W4A4
```

## How it was made

The experts and MLPs are NVFP4: E2M1 weights with one E4M3 scale per 16 inputs and an FP32 tensor scale,
activations quantized the same way at run time. Every attention projection is FP8 W8A8 (per-channel weights,
dynamic per-token activations): q/k/v of every layer and `o_proj` of every layer but 15. The routers, norms,
embeddings and vision tower stay in BF16; `lm_head` is tied to the embeddings. Weights were quantized with GPTQ
(FP4-aware MSE clip, router-weighted expert Hessians); the recipe was chosen by QDA's automated search under a
size budget equal to unsloth's checkpoint.

## Reproduce

Install QDA 0.1.0 (Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/Radioheading/qda.git && cd qda
uv venv -p 3.12 --managed-python && . .venv/bin/activate
uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
```

Then:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Gemma-4-26B-A4B-W4A4/*" --local-dir inputs
cd inputs/QDA-Gemma-4-26B-A4B-W4A4
sha256sum -c SHA256SUMS
qda build gemma-4-26b-a4b-w4a4-nvfp4.yaml --set out=QDA-Gemma-4-26B-A4B-W4A4-rebuild --set "devices=[0]"
```

Needs QDA 0.1.0 and one Blackwell GPU. Original model: `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, the revision the recipe builds from. `qda-kl kl qda-lab/QDA-Gemma-4-26B-A4B-W4A4` (its default prompts) reads KL 0.016756 to BF16 on a GB300. A rebuild's KL to BF16 is within ±5 % of this checkpoint's; on a GB300, a rebuild from this recipe was byte-identical to this checkpoint. The folder holds the recipe, the calibration data, `SHA256SUMS` over them and `CHECKPOINT.sha256` over this checkpoint; its `README.md` describes the calibration data and the evaluation setup. A rebuild on one GB300 takes about 10 min.

## License

Apache-2.0, the same as the base model ([Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).

## Citation

TBD.
