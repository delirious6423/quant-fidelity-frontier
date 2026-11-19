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
| AIME 2025 | 30 × 32 | 32,768 | 85.5 | 82.7 | 84.1 | 70.1 | 82.1 |
| HMMT Feb 2026 | 33 × 32 | 32,768 | 71.6 | 67.2 | 68.7 | 54.4 | 64.0 |
| GPQA-Diamond | 198 × 16 | 32,768 | 80.6 | 78.6 | 79.2 | 75.2 | 77.9 |
| IFBench | 300 × 16 | 32,768 | 80.6 | 79.9 | 80.9 | 78.1 | 79.3 |
| MMLU-Pro | 12,032 × 1 | 32,768 | 84.1 | 83.4 | 83.4 | 80.2 | 83.2 |
| LiveCodeBench | 454 × 4 | 32,768 | 81.3 | 80.2 | 79.5 | 70.9 | 77.5 |
| **Average** | | | 80.6 | 78.7 | 79.3 | 71.5 | 77.3 |

Thinking mode, pass@1 averaged over the draws. Same problems, seeds, sampling and output caps for every model.

The `<channel|>` marker is the end-of-thinking marker the harness splits each generation on.

## Usage

FP4 runs natively on NVIDIA Blackwell GPUs (B200, GB300, RTX PRO 6000); vLLM also serves it on H200. Evaluated with vLLM 0.33.0 on GB300.

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
qda build gemma-4-26b-a4b-w4a4-nvfp4.yaml --set out=QDA-Gemma-4-26B-A4B-W4A4-rebuild --set "devices=[0]"
```

Needs QDA 0.1.0 and one Blackwell GPU. Original model: `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, the revision the recipe builds from. On a GB300, a rebuild from this recipe was byte-identical to this checkpoint. A rebuild on one GB300 takes about 10 min.

## License

Apache-2.0, the same as the base model ([Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).

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
