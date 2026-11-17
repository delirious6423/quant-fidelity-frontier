---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [glm, glm-5.3, qda, quantized, nvfp4, fp4, w4a4, compressed-tensors, vllm, blackwell]
---

# QDA-GLM-5.3-Flash-W4A4

NVFP4 quantization (4-bit weights and 4-bit activations) of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) for NVIDIA Blackwell GPUs.
**197.2 GB vs. 328.3 GB for the original FP8 checkpoint, and 99.1 % of its accuracy** (average of four benchmarks).

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>FP8 (original)<br>328.3 GB</th>
<th>QDA W4A4<br>197.2 GB</th>
<th>NVIDIA NVFP4<br>204.4 GB</th>
<th>RedHatAI NVFP4<br>197.8 GB</th>
</tr></thead><tbody>
<tr><td>GPQA-Diamond<br><small>198 problems × 40 draws</small></td><td>90.08 ± 1.77</td><td><strong>89.84 ± 1.73</strong></td><td>88.89 ± 1.85</td><td>89.36 ± 1.79</td></tr>
<tr><td>Math<br><small>93 problems × 32 draws</small></td><td>83.90 ± 2.73</td><td><strong>82.39 ± 2.80</strong></td><td>81.25 ± 3.00</td><td>78.86 ± 3.13</td></tr>
<tr><td>LiveCodeBench v6<br><small>454 problems × 8 draws</small></td><td>71.26 ± 1.93</td><td><strong>70.48 ± 1.97</strong></td><td>69.93 ± 1.99</td><td>68.83 ± 2.00</td></tr>
<tr><td>SuperGPQA<br><small>9,000 problems × 2 draws</small></td><td>69.44 ± 0.46</td><td><strong>69.21 ± 0.46</strong></td><td>69.17 ± 0.46</td><td>69.01 ± 0.46</td></tr>
<tr><td><b>Average</b></td><td>78.7</td><td><strong>78.0</strong></td><td>77.3</td><td>76.5</td></tr>
<tr><td>Share of FP8</td><td>100 %</td><td><strong>99.1 %</strong></td><td>98.3 %</td><td>97.3 %</td></tr>
</tbody></table>

Accuracy in percent ± one standard error over problems, in percentage points: the standard deviation of each
problem's mean over its draws, divided by the square root of the number of problems. An answer that reaches the
65,536-token cap without finishing counts as wrong. Source: this release's four evaluation runs, one record per
problem and draw, graded by one grader for all four checkpoints; every checkpoint ran on the same problems,
draws and seeds. Comparison checkpoints: `zai-org/GLM-5.3-Flash` at revision `46a050b` (FP8),
`nvidia/GLM-5.3-Flash-NVFP4` at `1d00738`, `RedHatAI/GLM-5.3-Flash-NVFP4` at `18d55bf`. Sizes are the sum of
the `*.safetensors` files in decimal GB. The average is the arithmetic mean of the four benchmark accuracies.
Math is AIME 2026, HMMT Feb 2026 and HMMT Nov 2025.

### Evaluation settings

| Benchmark | Problems × draws | Output cap | Sampling |
|---|---|---|---|
| GPQA-Diamond | 198 × 40 | 65,536 tokens | temperature 1.0, top-p 0.95 |
| Math (AIME 2026, HMMT Feb 2026, HMMT Nov 2025) | 93 × 32 | 65,536 tokens | temperature 1.0, top-p 0.95 |
| LiveCodeBench v6 | 454 × 8 | 65,536 tokens | temperature 1.0, top-p 0.95 |
| SuperGPQA | 9,000 × 2 | 65,536 tokens | temperature 1.0, top-p 0.95 |

- Thinking mode on, with each checkpoint's own chat template and no template overrides; one user message per
  problem.
- Seeds: one per problem and draw, identical for all four checkpoints.
- Server: vLLM 0.31.0, tensor parallelism 4 on 4× GB300, engine window 350,000 tokens, at most 256 concurrent
  sequences, GPU memory utilization 0.90, MTP speculative decoding with 5 tokens, and the `glm47` reasoning
  parser. KV cache in BF16 for every checkpoint.
- Image and video inputs were disabled (`--limit-mm-per-prompt {"image":0,"video":0}`); every benchmark is
  text-only.

Share of answers cut off at the 65,536-token limit (source: the same four runs as the table above; lower is
better):

| Benchmark | FP8 | QDA W4A4 | NVIDIA | RedHatAI |
|---|---:|---:|---:|---:|
| GPQA-Diamond | 3.08 % | 3.83 % | 4.85 % | 4.49 % |
| Math | 14.28 % | 15.39 % | 17.41 % | 19.42 % |
| LiveCodeBench v6 | 25.03 % | 26.10 % | 26.76 % | 27.97 % |
| SuperGPQA | 1.59 % | 1.88 % | 1.94 % | 2.58 % |

QDA − NVIDIA, paired over the same problems and draws, with a 95 % interval over problems: GPQA-Diamond +0.95
(+0.24 to +1.65), Math +1.14 (−0.38 to +2.66), LiveCodeBench v6 +0.55 (−0.34 to +1.44), SuperGPQA +0.04
(−0.43 to +0.52). An earlier, independent set of 20 GPQA-Diamond draws had QDA 0.7 points behind NVIDIA, so we
read this as: not behind NVIDIA's checkpoint on any of the four benchmarks, at a 7 GB smaller size; it is not a
clear win.

## Usage

Needs 4 NVIDIA Blackwell data-center GPUs (B200, B300, GB200 or GB300); evaluated with vLLM 0.31.0.

```bash
vllm serve qda-lab/QDA-GLM-5.3-Flash-W4A4 --tensor-parallel-size 4 --reasoning-parser glm47 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'
```

## Format

The routed experts of the MoE layers (3–44) and the dense MLPs of layers 0–2 are NVFP4: E2M1 weights with one
E4M3 scale per 16 inputs and an FP32 tensor scale, and activations quantized the same way at run time. The MTP
layer's experts are FP8 (block-scaled weights, dynamic activation scales). Attention (linear and sparse MLA),
shared experts, routers, embeddings, `lm_head` and the vision tower stay in BF16. The checkpoint is in the
compressed-tensors format. The quantization recipe is not published.

## License

MIT, the same as the base model.
