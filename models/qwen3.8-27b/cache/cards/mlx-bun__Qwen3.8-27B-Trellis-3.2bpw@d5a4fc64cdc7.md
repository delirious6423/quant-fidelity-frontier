---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx-bun
tags:
- mlx
- mlx-bun
- apple-silicon
- quantized
- trellis
- qtip
- 3-bit
model-index:
- name: Qwen3.8-27B-Trellis-3.2bpw
  results:
  - task:
      type: text-generation
      name: Mathematical reasoning
    dataset:
      type: madrylab/gsm8k-platinum
      name: GSM8K-Platinum
      config: main
      split: test
    metrics:
    - type: exact_match
      name: Strict match (0-shot, seed 1234)
      value: 96.11
    - type: exact_match
      name: Flexible extraction (0-shot, seed 1234)
      value: 96.28
  - task:
      type: text-generation
      name: Instruction following
    dataset:
      type: google/IFEval
      name: IFEval
      split: train
    metrics:
    - type: accuracy
      name: Prompt-level strict accuracy
      value: 91.68
    - type: accuracy
      name: Instruction-level strict accuracy
      value: 94.48
  - task:
      type: text-generation
      name: Graduate-level reasoning
    dataset:
      type: Idavidrein/gpqa
      name: GPQA Diamond
      config: gpqa_diamond
      split: train
    metrics:
    - type: accuracy
      name: Pass@1 (0-shot, seed 1234)
      value: 72.73
---

# Qwen3.8-27B-Trellis-3.2bpw

## Model Overview

A **3.23 bpw quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** for local inference on Apple Silicon with [mlx-bun](https://github.com/joshuarossi/mlx-bun).

- **27 billion parameters**, compressed to **11.99 GB** from approximately 54 GB in BF16.
- **Mixed 2/3/4-bit Trellis quantization**, with affine 3/4-bit text projections and the original BF16 vision encoder.
- **MTP companion included:** 259 MB in `mtp/`, bringing the complete download to **12.25 GB**.
- **Runtime:** mlx-bun's Metal kernels. This packed format requires mlx-bun.
- **License:** Apache 2.0.

[Quantization recipe](QUANTIZATION.md) · [MTP companion](mtp/README.md)

## Benchmark Results

### Benchmark scores

| Benchmark | Score | Correct / total |
|---|---:|---:|
| GSM8K-Platinum · flexible extraction | **96.28%** | 1,164 / 1,209 |
| IFEval · prompt-level strict | **91.68%** | 496 / 541 |
| GPQA Diamond · pass@1 | **72.73%** | 144 / 198 |

Full datasets, zero-shot, thinking enabled, seed 1234, MTP off. GSM8K strict exact match is **96.11%**. Thirteen GPQA responses reached the token limit and scored incorrect.

### Generation speed

**Apple M1 Max, 32 GB unified memory**

| Configuration | Tokens per second |
|---|---:|
| Single request, MTP off | **10.76** |
| Single request, MTP depth 2 | **14.19** |
| Batch 4, 4-bit KV, MTP off | **18.00 total** |

Single-request speeds are medians of five 192-token generations with a 757-token prompt and default KV precision. Batch speed uses four concurrent 128-token generations with short prompts. Long reasoning throughput varies with context length.

[Evaluation protocol and full performance suite](eval/README.md) · [GSM8K](eval/gsm8k-platinum.json) · [IFEval](eval/ifeval.json) · [GPQA](eval/gpqa-diamond.json)

## Accuracy Compared with BF16

| Benchmark | BF16 base¹ | This quant | Difference |
|---|---:|---:|---:|
| GSM8K-Platinum · strict exact match | 96.25% | 96.11% | **−0.14 points** |
| IFEval · prompt-level strict | 91.19% | 91.68% | **+0.49 points** |
| GPQA Diamond · pass@1 | 89.56% | 72.73% | **−16.83 points** |

The quantized model achieves **81.2% of the published BF16 GPQA score** with approximately **78% less model storage**. GSM8K-Platinum and IFEval scores differ from the published BF16 results by less than one percentage point.

¹ Baselines and approximate BF16 size come from [Red Hat's published evaluation](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4#evaluation). Red Hat reports three-seed averages; ours use one seed, mlx-bun, and different KV settings. The differences compare complete inference configurations rather than isolating weight quantization alone. Vision quality has not been separately evaluated for this quant.

<details>
<summary><strong>How to run it</strong></summary>

Install [mlx-bun](https://mlx-bun.dev/getting-started/installation/), then:

```sh
mlx-bun get mlx-bun/Qwen3.8-27B-Trellis-3.2bpw
mlx-bun serve mlx-bun/Qwen3.8-27B-Trellis-3.2bpw \
  --kv-quant 4 --batch 4 \
  --ssd-cache "$HOME/.cache/mlx-bun/trellis-kv" \
  --ssd-cache-max 20 --generation-checkpoint 1024
```

- OpenAI-compatible API: `http://localhost:8080/v1`.
- For MTP, add `--draft-kind mtp --num-draft-tokens 2`; the bundled companion loads automatically.
- Thinking is enabled by default. Sampling: `temperature=1.0`, `top_p=0.95`, `top_k=20`.
- Disk size is separate from inference memory; KV cache and activations require additional RAM.
- Evaluated runtime: [`refactor/monorepo` at `7c6155b7`](https://github.com/joshuarossi/mlx-bun/commit/7c6155b7d49632f425788e04784d54b6ae93d327).

</details>

[Apache 2.0](LICENSE) · Base model by [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) · Quantization and runtime by [mlx-bun](https://github.com/joshuarossi/mlx-bun) · Trellis method based on [QTIP](https://arxiv.org/abs/2406.11235).
