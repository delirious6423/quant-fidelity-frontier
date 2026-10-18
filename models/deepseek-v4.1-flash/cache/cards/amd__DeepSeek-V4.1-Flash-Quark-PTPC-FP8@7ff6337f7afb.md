---
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - ptpc-fp8
  - quantization
  - amd-quark
  - moe
  - deepseek
---
## Model Overview

- **Model Architecture:** DeepseekV41ForCausalLM
  - **Input:** Text, Image
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI355 / MI350 (gfx950)
- **ROCm:** 7.2.0
- **PyTorch:** 2.12.0
- **Transformers:** 5.17.0
- **Operating System(s):** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html) (v0.13.0)
  - **Quantized layers:** `experts`, `shared_experts`, `self_attn`, and `engram.embed` are quantized to PTPC-FP8. The vision part is not quantized.

## Model Quantization

Quantized from [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with [AMD Quark](https://quark.docs.amd.com/latest/index.html). The source MXFP4
and MXFP8 weights are converted to PTPC-FP8: `experts`,
`shared_experts`, `self_attn`, and `engram.embed` are quantized with PTPC-FP8.
The vision tower is kept in its original precision.

### Quantization script

```bash
MODEL_DIR="deepseek-ai/DeepSeek-V4.1-Flash"
OUTPUT_DIR="amd/DeepSeek-V4.1-Flash-Quark-PTPC-FP8"

HIP_VISIBLE_DEVICES=0 \
QUARK_MXFP4_IMPL=triton \
python3 Quark/examples/torch/language_modeling/llm_ptq/quantize_quark.py \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --quant_scheme ptpc_fp8 \
  --layer_quant_scheme "*.engram.embed" ptpc_fp8 \
  --quant_flow file2file \
  --model_export hf_format \
  --keep_excluded_layers_as_original_model_state \
  --exclude_layers \
    "embed" \
    "*head*" \
    "*ffn.gate*" \
    "*attn.compressor*" \
    "*attn.indexer.weights_proj*" \
    "*attn.indexer.wk*" \
    "*hc_*" \
    "*vision*" \
    "*aligner*"
```

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend based on the Docker image `vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`.

Required vLLM patches: [#57760](https://github.com/vllm-project/vllm/pull/57760), [#57071](https://github.com/vllm-project/vllm/pull/57071).

## Evaluation

The model was evaluated on GSM8K (8-shot) using the vLLM framework.

### Accuracy

| Benchmark | deepseek-ai/DeepSeek-V4.1-Flash | amd/DeepSeek-V4.1-Flash-Quark-PTPC-FP8 (this model) | Recovery |
|---|---:|---:|---:|
| GSM8K (8-shot, flexible-extract) | 92.95 | 92.95 | 100.0% |

GSM8K is measured with lm-eval. The base and quantized models are run with the
same framework, settings, and host.

### Reproduction

The evaluation runs against a vLLM server, using the Docker image
`vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`.

#### Launching server
```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MHA=1
export SAFETENSORS_FAST_GPU=1
export VLLM_MOE_SKIP_PADDING=0
vllm serve amd/DeepSeek-V4.1-Flash-Quark-PTPC-FP8 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --moe-backend aiter \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8008
```

#### Evaluating GSM8K in a new terminal
```bash
lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args "model=amd/DeepSeek-V4.1-Flash-Quark-PTPC-FP8,base_url=http://127.0.0.1:8008/v1/completions,num_concurrent=32,max_retries=5,timeout=1800,tokenized_requests=False,tokenizer_backend=None,max_length=8192" \
  --gen_kwargs "max_gen_toks=4096" \
  --num_fewshot 8 \
  --log_samples \
  --output_path /tmp/gsm8k-dsv41-ptpc-fp8
```

## License

This model is a quantized derivative of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
and is distributed under the same license as the source model: the
[MIT License](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE).
A copy of the upstream LICENSE is included in this repository.

Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
AMD has modified selected language-model weights by quantizing them to PTPC-FP8
with AMD Quark; the modifications are provided under the same MIT License and
are not subject to any separate or different license.
