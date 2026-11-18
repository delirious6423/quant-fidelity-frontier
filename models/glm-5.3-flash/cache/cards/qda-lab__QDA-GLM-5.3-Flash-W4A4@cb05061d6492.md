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
**197.2 GB vs. 328.3 GB for the original FP8 checkpoint, and 98.3 % of its accuracy** (average of four benchmarks).

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>FP8 (original)<br>328.3 GB</th>
<th>QDA W4A4<br>197.2 GB</th>
<th>NVIDIA NVFP4<br>204.4 GB</th>
<th>RedHatAI NVFP4<br>197.8 GB</th>
</tr></thead><tbody>
<tr><td>GPQA-Diamond<br><small>198 problems × 12 draws</small></td><td>86.07 ± 2.16</td><td>83.88 ± 2.27</td><td>83.33 ± 2.32</td><td><strong>84.01 ± 2.25</strong></td></tr>
<tr><td>Math<br><small>93 problems × 16 draws</small></td><td>84.81 ± 2.72</td><td><strong>83.27 ± 2.79</strong></td><td>80.91 ± 3.13</td><td>78.83 ± 3.18</td></tr>
<tr><td>LiveCodeBench v6<br><small>454 problems × 4 draws</small></td><td>71.31 ± 2.00</td><td><strong>70.37 ± 2.01</strong></td><td>69.33 ± 2.06</td><td>69.16 ± 2.02</td></tr>
<tr><td>SuperGPQA<br><small>9,000 problems × 1 draw</small></td><td>68.66 ± 0.49</td><td><strong>67.91 ± 0.49</strong></td><td>67.77 ± 0.49</td><td>67.38 ± 0.49</td></tr>
<tr><td><b>Average</b></td><td>77.7</td><td><strong>76.4</strong></td><td>75.3</td><td>74.8</td></tr>
<tr><td>Share of FP8</td><td>100 %</td><td><strong>98.3 %</strong></td><td>96.9 %</td><td>96.3 %</td></tr>
</tbody></table>

Accuracy in percent ± one standard error over problems, in percentage points: the standard deviation of each
problem's mean over its draws, divided by the square root of the number of problems. An answer that reaches the
output cap of its benchmark (next table) without finishing counts as wrong. Source: this release's four evaluation
runs, one record per problem and draw, graded by one grader for all four checkpoints; every checkpoint ran on the
same problems, draws and seeds. Comparison checkpoints: `zai-org/GLM-5.3-Flash` at revision `46a050b` (FP8),
`nvidia/GLM-5.3-Flash-NVFP4` at `1d00738`, `RedHatAI/GLM-5.3-Flash-NVFP4` at `18d55bf`. Sizes are the sum of
the `*.safetensors` files in decimal GB. The average is the arithmetic mean of the four benchmark accuracies.
Math is AIME 2026, HMMT Feb 2026 and HMMT Nov 2025.

### Evaluation settings

| Benchmark | Problems × draws | Output cap | Sampling |
|---|---|---|---|
| GPQA-Diamond | 198 × 12 | 32,768 tokens | temperature 1.0, top-p 0.95 |
| Math (AIME 2026, HMMT Feb 2026, HMMT Nov 2025) | 93 × 16 | 65,536 tokens | temperature 1.0, top-p 0.95 |
| LiveCodeBench v6 | 454 × 4 | 65,536 tokens | temperature 1.0, top-p 0.95 |
| SuperGPQA | 9,000 × 1 | 32,768 tokens | temperature 1.0, top-p 0.95 |

- Thinking mode on, with each checkpoint's own chat template and no template overrides; one user message per
  problem.
- Seeds: one per problem and draw, identical for all four checkpoints.
- Server: vLLM 0.31.0, tensor parallelism 4 on 4× GB300, engine window 163,840 tokens, at most 256 concurrent
  sequences, GPU memory utilization 0.90, MTP speculative decoding with 5 tokens, and the `glm47` reasoning
  parser. KV cache in BF16 for every checkpoint.
- Image and video inputs were disabled (`--limit-mm-per-prompt {"image":0,"video":0}`); every benchmark is
  text-only.

Share of answers cut off at the output cap of their benchmark (source: the same four runs as the table above; lower is
better):

| Benchmark | FP8 | QDA W4A4 | NVIDIA | RedHatAI |
|---|---:|---:|---:|---:|
| GPQA-Diamond | 8.71 % | 11.20 % | 12.37 % | 11.74 % |
| Math | 13.17 % | 15.32 % | 17.00 % | 19.96 % |
| LiveCodeBench v6 | 25.50 % | 25.88 % | 27.48 % | 27.97 % |
| SuperGPQA | 4.07 % | 4.43 % | 5.03 % | 5.61 % |

QDA − NVIDIA, paired over the same problems and draws, with a 95 % interval over problems: GPQA-Diamond +0.55
(−0.54 to +1.64), Math +2.35 (+0.10 to +4.61), LiveCodeBench v6 +1.05 (−0.15 to +2.24), SuperGPQA +0.14
(−0.52 to +0.81).

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
