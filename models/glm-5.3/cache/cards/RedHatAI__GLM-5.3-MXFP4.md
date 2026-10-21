---
language:
- en
- zh
library_name: transformers
license: other
license_name: glm-5.3
pipeline_tag: text-generation
tags:
- mxfp4
- fp4
- vllm
- llm-compressor
- compressed-tensors
base_model: zai-org/GLM-5.3
---

# GLM-5.3-MXFP4

## Model Overview
- **Model Architecture:** GlmMoeDsaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 2026-09-15
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3).
It was evaluated on several tasks to assess its quality in comparison to the FP8 checkpoint.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) to FP4 (MXFP4, OCP Microscaling) data type, ready for inference with vLLM. Weights use FP4 (e2m1) with group size 32 and power-of-2 (E8M0) scales, while activations use FP4 with dynamic per-group scaling.

This optimization reduces the number of bits per parameter from 8 (FP8) to 4, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/GLM-5.3-MXFP4 \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 4 \
  --reasoning-parser glm45 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with the MXFP4 scheme in model-free post-training quantization (no calibration data required), exported in compressed-tensors format.

<details>

```python
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "zai-org/GLM-5.3"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*eh_proj$",
    "re:.*self_attn.indexer.weights_proj$",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="MXFP4",
    ignore=ignore + ["re:.*self_attn.indexer.wk$"],
    converter=FP8BlockDequantizer(ignore=ignore),
    max_workers=2,
    device="cuda:0",
)
```

</details>

## Evaluation

This model was evaluated on GSM8K Platinum, IFEval, MATH-500, AIME 2025, and GPQA Diamond using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness) and [lighteval](https://github.com/neuralmagic/lighteval), all served with vLLM (OpenAI-compatible API). Each benchmark was run with 3 seeds (1234, 2345, 3456) and the results averaged; recovery is computed against the FP8 checkpoint on 4 B200s.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>zai-org/GLM-5.3 (FP8)</th>
      <th>RedHatAI/GLM-5.3-MXFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Instruction Following</b></td>
      <td>IFEval (prompt-level strict)</td>
      <td>91.87%</td>
      <td>91.19%</td>
      <td>99.26%</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Reasoning</b></td>
      <td>GSM8K Platinum (strict-match)</td>
      <td>97.60%</td>
      <td>96.36%</td>
      <td>98.73%</td>
    </tr>
    <tr>
      <td>MATH-500 (pass@1)</td>
      <td>95.67%</td>
      <td>94.00%</td>
      <td>98.26%</td>
    </tr>
    <tr>
      <td>AIME 2025 (pass@1)</td>
      <td>98.33%</td>
      <td>91.25%</td>
      <td>92.80%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (pass@1)</td>
      <td>92.42%</td>
      <td>89.06%</td>
      <td>96.36%</td>
    </tr>
  </tbody>
</table>
