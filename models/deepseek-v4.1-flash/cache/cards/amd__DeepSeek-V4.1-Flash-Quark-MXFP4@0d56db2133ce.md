---
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - ocp-mx
  - quantization
  - amd-quark
  - moe
  - deepseek
---

# DeepSeek-V4.1-Flash-MXFP4

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
  - **Quantized layers:** `experts`, `shared_experts`, and `self_attn` in language model. The vision part is not quantized.
  - **experts and shared_experts:** OCP MXFP4 for both weights and activations
  - **self_attn:** OCP MXFP8 for both weights and activations

## Model Quantization

Quantized from [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with [AMD Quark](https://quark.docs.amd.com/latest/index.html). The MoE expert
projections (`experts`, `shared_experts`) are quantized to OCP MXFP4 and the
language-model attention (`self_attn`) to OCP MXFP8 — weights and activations in
both cases. The vision tower and the remaining modules (router gate, embeddings,
output head, engram and MTP blocks) are kept in their original precision.

### Quantization script

```python
from quark.torch import LLMTemplate, ModelQuantizer

MODEL_DIR = "deepseek-ai/DeepSeek-V4.1-Flash"
OUTPUT_DIR = "amd/DeepSeek-V4.1-Flash-MXFP4"

template = LLMTemplate(
    model_type="deepseek_v41",
    kv_layers_name=["*wkv"],
    q_layer_name=["*wq_a", "*wq_b"],
    exclude_layers_name=[
        "*attn*",
        "embed",
        "*head*",
        "*ffn.gate*",
        "hc_*",
        "*engram*",
        "*vision*",
        "*aligner*",
        "*main_proj*",
    ],
)

LLMTemplate.register_template(template)

quant_config = template.get_config(scheme="mxfp4")

quantizer = ModelQuantizer(quant_config)
quantizer.direct_quantize_checkpoint(
    pretrained_model_path=MODEL_DIR,
    save_path=OUTPUT_DIR,
    keep_excluded_layers_as_original_model_state=True,
)
```

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend based on the Docker image `vllm/vllm-openai-rocm:deepseekv41-flash-0909`.

## Evaluation

The model was evaluated on GSM8K (8-shot) and GPQA Diamond (0-shot) using the vLLM framework.

### Accuracy

| Benchmark | deepseek-ai/DeepSeek-V4.1-Flash | amd/DeepSeek-V4.1-Flash-MXFP4 (this model) | Recovery |
|---|---:|---:|---:|
| GSM8K (8-shot, flexible-extract) | 92.87 | 92.34 | 99.4% |
| GPQA Diamond (0-shot, thinking) | 89.39 | 90.40 | 101.1% |

GSM8K is measured with lm-eval and GPQA Diamond with sgl-eval. For each benchmark
the base and quantized models are run with the same framework, settings, and host.

GPQA Diamond is scored over all 198 questions using stochastic sampling
(`temperature=1.0, top_p=0.95`), so scores vary from run to run; the 95% confidence
interval on 198 questions is ±2.33%. On this draw the quantized model answered
179/198 correctly against 177/198 for the base model. That gap is well inside the
confidence interval, so the two should be read as equivalent rather than as the
quantized model outperforming the base.

### Reproduction

The evaluation runs against a vLLM server, using the Docker image
`vllm/vllm-openai-rocm:deepseekv41-flash-0909`.

#### Launching server
```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_MOE=0
vllm serve amd/DeepSeek-V4.1-Flash-MXFP4 --tensor-parallel-size 4 \
  --trust-remote-code --tokenizer-mode deepseek_v41 \
  --max-model-len 73728 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager
```

The context length must accommodate GPQA Diamond's 65536-token generation budget
in thinking mode; GSM8K alone runs comfortably at `--max-model-len 8192`.

#### Evaluating GSM8K in a new terminal
```bash
lm_eval --model local-completions \
    --model_args model=amd/DeepSeek-V4.1-Flash-MXFP4,base_url=http://localhost:8000/v1/completions,tokenized_requests=False,num_concurrent=8,tokenizer=amd/DeepSeek-V4.1-Flash-MXFP4 \
    --tasks gsm8k --batch_size auto --num_fewshot 8 --seed 42 \
    --gen_kwargs "temperature=0,max_gen_toks=4096"
```

#### Evaluating GPQA Diamond
```bash
pip install sgl-eval
sgl-eval run gpqa \
    --model amd/DeepSeek-V4.1-Flash-MXFP4 \
    --api-key EMPTY \
    --base-url http://localhost:8000/v1 \
    --num-threads 8 \
    --n-repeats 1 \
    --max-tokens 65536 \
    --temperature 1.0 \
    --top-p 0.95 \
    --thinking
```

## License

This model is a quantized derivative of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
and is distributed under the same license as the source model: the
[MIT License](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE).
A copy of the upstream LICENSE is included in this repository.

Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
AMD has modified the model weights of the MoE expert layers by quantizing them to
MXFP4 with AMD Quark; the modifications are provided under the same MIT License and
are not subject to any separate or different license.
