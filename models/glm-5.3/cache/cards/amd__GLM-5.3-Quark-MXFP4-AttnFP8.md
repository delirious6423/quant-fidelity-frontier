---
license: other
license_name: glm-5.3
base_model:
- zai-org/GLM-5.3
language:
- en
- zh
pipeline_tag: text-generation
library_name: transformers
---
# Model Overview

- **Model Architecture:** GlmMoeDsaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI350/MI355
- **ROCm:** 7.1.1
- **PyTorch:** 2.10.0+rocm7.1.1.lw.gitd9556b05
- **Transformers:** 5.15.1
- **Operating System(s):** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html) (V0.12)
  - **Quantized layers:** self_attn, router experts, shared_experts 
  - **self_attn:** FP8 per-block
  - **Router experts and shared_experts:** MXFP4

This model was built with GLM-5.3 model by applying [AMD-Quark](https://quark.docs.amd.com/latest/index.html) for MXFP4 quantization.

# Model Quantization

The model was quantized from [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) using [AMD-Quark](https://quark.docs.amd.com/latest/index.html). The weights and activations are quantized to MXFP4.

**Quantization scripts:**

```bash
cd Quark/examples/torch/language_modeling/llm_ptq/
python quantize_quark.py \
  --model_dir zai-org/GLM-5.3 \
  --output_dir amd/GLM-5.3-Quark-MXFP4 \
  --quant_scheme mxfp4 \
  --exclude_layers \
    "*self_attn*" \
    "*lm_head" \
    "*embed_tokens" \
    "*eh_proj" \
    "*mlp.gate" \
  --file2file_quantization \
  --keep_excluded_layers_as_original_model_state
```

# Deployment
### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backends.

## Evaluation
The model was evaluated on GSM8K benchmarks. 

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>GLM-5.3 </strong>
   </td>
   <td><strong>GLM-5.3-Quark-MXFP4(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>GSM8K (flexible-extract)
   </td>
   <td>91.96
   </td>
   <td>91.43
   </td>
   <td>99.42%
   </td>
  </tr>
</table>

### Reproduction

The GSM8K results were obtained using the `lm-evaluation-harness` framework, based on the Docker image `vllm/vllm-openai-rocm:v0.31.0`.

```bash
pip install lm-eval[api]

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_ROCM_USE_AITER_FP4BMM=0

lm_eval --model vllm \
  --model_args "pretrained=amd/GLM-5.3-Quark-MXFP4,tensor_parallel_size=4,dtype=auto,quantization=quark,max_model_len=32768,trust_remote_code=True,max_num_seqs=32" \
  --tasks gsm8k \
  --batch_size auto
```

# License
Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.