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
<tr><td>GPQA-Diamond<br><small>198 problems × 12 draws</small></td><td>86.07</td><td>83.88</td><td>83.33</td><td><strong>84.01</strong></td></tr>
<tr><td>Math<br><small>93 problems × 16 draws</small></td><td>84.81</td><td><strong>83.27</strong></td><td>80.91</td><td>78.83</td></tr>
<tr><td>LiveCodeBench v6<br><small>454 problems × 4 draws</small></td><td>71.31</td><td><strong>70.37</strong></td><td>69.33</td><td>69.16</td></tr>
<tr><td>SuperGPQA<br><small>9,000 problems × 1 draw</small></td><td>68.66</td><td><strong>67.91</strong></td><td>67.77</td><td>67.38</td></tr>
<tr><td><b>Average</b></td><td>77.7</td><td><strong>76.4</strong></td><td>75.3</td><td>74.8</td></tr>
<tr><td>Share of FP8</td><td>100 %</td><td><strong>98.3 %</strong></td><td>96.9 %</td><td>96.3 %</td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over the draws; same problems, draws and seeds for every checkpoint. Math is AIME 2026, HMMT Feb 2026 and HMMT Nov 2025. Served with vLLM 0.31.0 on 4× GB300 (tensor parallel 4).

## Usage

Needs 4 NVIDIA Blackwell GPUs.

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
