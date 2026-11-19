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
| AIME 2025 | 30 × 32 | 81,920 | 88.4 | 87.1 | 91.1 | 87.1 | 88.9 |
| HMMT Feb 2026 | 33 × 32 | 81,920 | 72.6 | 73.4 | 74.9 | 76.8 | 72.5 |
| GPQA-Diamond | 198 × 16 | 32,768 | 82.4 | 80.8 | 81.1 | 80.2 | 81.7 |
| IFBench | 300 × 16 | 32,768 | 67.0 | 68.0 | 68.6 | 67.6 | 68.7 |
| MMLU-Pro | 12,032 × 1 | 32,768 | 83.6 | 83.1 | 83.4 | 83.2 | 83.2 |
| LiveCodeBench | 454 × 4 | 81,920 | 86.7 | 85.2 | 85.8 | 87.3 | 85.5 |
| **Average** | | | 80.1 | 79.6 | 80.8 | 80.4 | 80.1 |

Thinking mode, pass@1 averaged over the draws; sampling temperature 1.0, top-p 0.95, top-k 20. Same problems, seeds, sampling and output caps for every model.

### Three further math sets

BF16, QDA W4A4 and NVIDIA were also run on three further MathArena sets, asked and graded the same way as the first two:

| | AIME 2025 | HMMT Feb 2026 | AIME 2026 | HMMT Nov 2025 | HMMT Feb 2025 | all 153 problems |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 88.4 | 72.6 | 87.9 | 79.2 | 78.0 | 81.1 |
| QDA W4A4 | 87.1 | 73.4 | 89.2 | 79.9 | 82.7 | 82.3 |
| NVIDIA | 91.1 | 74.9 | 89.3 | 77.9 | 80.9 | 82.7 |

The last column pools all 153 problems.

## Usage

FP4 runs natively on NVIDIA Blackwell GPUs (B200, GB300, RTX PRO 6000); vLLM also serves it on H200. Evaluated with vLLM 0.33.0 on GB300.

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
uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
```

Then:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W4A4/*" --local-dir inputs
cd inputs/QDA-Qwen3.8-27B-W4A4
qda build qwen3.8-27b-w4a4-nvfp4.yaml --set out=QDA-Qwen3.8-27B-W4A4-rebuild --set "devices=[0]"
```

Needs QDA 0.1.0 and one Blackwell GPU. Original model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the revision the recipe builds from. On a GB300, a rebuild from this recipe was byte-identical to this checkpoint. A rebuild on one GB300 takes about 23 min.

## License

Apache-2.0, the same as the base model.

## Citation

[QDA: Closed-Loop LLM Quantization Design with Agents](https://arxiv.org/abs/2611.23388)

```bibtex
@article{li2026qda,
  title   = {QDA: Closed-Loop LLM Quantization Design with Agents},
  author  = {Li, Xingyang and Zhao, Tianlang and Han, Song and Zhu, Ligeng},
  journal = {arXiv preprint arXiv:2611.23388},
  year    = {2026}
}
```
