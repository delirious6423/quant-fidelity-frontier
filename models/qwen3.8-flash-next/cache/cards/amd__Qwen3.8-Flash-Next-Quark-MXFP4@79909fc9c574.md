---
base_model:
- Qwen/Qwen3.8-Flash-Next
language:
- en
library_name: transformers
license: other
license_link: https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4/blob/main/LICENSE
---

# Model Overview

- **Model Architecture:** Qwen4ExpForConditionalGeneration
  - **Input:** Text, Image, Video
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI350 / MI355
- **ROCm**: 7.2.3
- **PyTorch**: 2.12.0+git6bbd260
- **Transformers**: 5.16.1
- **Operating System(s):** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html)
  - **Quantized layers:** All MoE experts in the language model, **including the shared expert**. Attention is kept in bf16.
  - **Weight quantization:**
    - MOE OCP MXFP4, Static
  - **Activation quantization:**
    - MOE OCP MXFP4, Dynamic

This checkpoint quantizes the routed **and** shared MoE experts to OCP MXFP4 and keeps everything
else in bf16, including the full self-attention and the (Mamba-style) linear-attention
projections, the MTP head, router gates, shared-expert gate, PLE n-gram path, vision tower,
hyper-connection mixers, and `lm_head`.

# Model Quantization

The model was quantized from [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) using [AMD-Quark](https://quark.docs.amd.com/latest/index.html). MoE expert weights and
activations are quantized to OCP MXFP4.

**Quantization script:**

```python
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.quantization.weight_convert import SplitFusedExperts, WeightConverter

EXCLUDE = [
    "lm_head", "*mlp.gate", "*shared_expert_gate", "mtp.*", "model.visual.*",
    "*hyper_connection*", "*ple.*", "*linear_attn.*", "*self_attn.*",
]

template = LLMTemplate(
    model_type="qwen4_exp",
    kv_layers_name=["*self_attn.k_proj", "*self_attn.v_proj"],
    q_layer_name="*self_attn.q_proj",
    exclude_layers_name=EXCLUDE,
    f2f_weight_converters=[
        WeightConverter("gate_up_proj", ["gate_proj.weight", "up_proj.weight"],
                        operations=[SplitFusedExperts(split_axis=0)]),
        WeightConverter("down_proj", ["down_proj.weight"],
                        operations=[SplitFusedExperts(split_axis=0)]),
    ],
)
LLMTemplate.register_template(template)

quant_config = template.get_config(scheme="mxfp4", exclude_layers=EXCLUDE)

quantizer = ModelQuantizer(quant_config)
quantizer.direct_quantize_checkpoint(
    pretrained_model_path="Qwen/Qwen3.8-Flash-Next",
    save_path=OUTPUT_DIR,
    weight_converters=template.f2f_weight_converters,
    device="cuda",
)
```

For further details or issues, please refer to the AMD-Quark documentation.

# Evaluation

The model was evaluated using [vLLM](https://docs.vllm.ai/en/latest/), [EvalScope](https://evalscope.readthedocs.io/en/latest/), and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Accuracy

| **Benchmark** | **Qwen/Qwen3.8-Flash-Next** | **amd/Qwen3.8-Flash-Next-Quark-MXFP4 (this model)** | **Recovery** |
| --- | --- | --- | --- |
| GSM8K (5-shot) | 96.8 | 96.5 | 99.7% |
| GPQA Diamond (mean@8) | 91.67 | 91.67 | 100.00% |
| AIME26 (mean@16) | 99.17 | 98.33 | 99.15% |
| MMMU Pro, standard (mean@1) | 79.65 | 80.00 | 100.44% |
| MMMU Pro, vision (mean@1) | 78.38 | 77.40 | 98.75% |
| SciCode, subproblem (mean@8) | 49.01 | 48.58 | 99.12% |
| AA-LCR (mean@16) | 78.50 | 80.63 | 102.71% |
| IFBench (mean@5) | 83.27 | 82.67 | 99.28% |

Note: Results may vary from run to run.

### Reproduction

The following results were obtained using vLLM and EvalScope `1.14.1`. SciCode uses sandbox image `scicode-benchmark:7687150b169df5091c158abb`.

#### Launching server

We launch the vLLM server following [vLLM Recipes](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next?hardware=mi355x&variant=bf16&frontend=rust) using container image `vllm/vllm-openai-rocm:nightly-83c078d955cd6e0c63d3b12abe669c8de936a370`.

```bash
VLLM_USE_RUST_FRONTEND=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
vllm serve amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gdn-prefill-backend aiter_flydsl \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```


#### Evaluating model in a new terminal

**GSM8K**

```bash
lm_eval --model local-chat-completions \
    --model_args "model=amd/Qwen3.8-Flash-Next-Quark-MXFP4,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=32,timeout=1200,max_retries=3,tokenized_requests=False,max_gen_toks=2048" \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template
```

**GPQA Diamond**

```bash
evalscope eval \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --datasets gpqa_diamond \
  --eval-batch-size 8 \
  --repeats 8 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":131072,"timeout":7200}' \
  --work-dir logs/gpqa_diamond
```

**AIME26**

```bash
evalscope eval \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --datasets aime26 \
  --eval-batch-size 8 \
  --repeats 16 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":131072,"timeout":7200}' \
  --work-dir logs/aime26
```

**MMMU Pro (standard & vision)**

```bash
evalscope eval \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --datasets mmmu_pro \
  --eval-batch-size 8 \
  --repeats 1 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":98304,"timeout":7200}' \
  --dataset-args '{"mmmu_pro":{"dataset_format":"standard (10 options)"}}' \
  --work-dir logs/mmmu_pro_standard

evalscope eval \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --datasets mmmu_pro \
  --eval-batch-size 8 \
  --repeats 1 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":98304,"timeout":7200}' \
  --dataset-args '{"mmmu_pro":{"dataset_format":"vision"}}' \
  --work-dir logs/mmmu_pro_vision
```

**SciCode**

```bash
evalscope eval \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --datasets scicode \
  --dataset-hub modelscope \
  --eval-batch-size 8 \
  --repeats 8 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":131072,"timeout":7200}' \
  --dataset-args '{"scicode":{"extra_params":{"provide_background":true}}}' \
  --sandbox '{"enabled":true,"engine":"docker","pool_size":4}' \
  --work-dir logs/scicode
```

**AA-LCR (v1.1)**

Judge configuration (aa-lcr-judge.json; replace `<YOUR_SUBSCRIPTION_KEY>`):

```json
{
  "strategy": "llm",
  "models": [
    {
      "model_id": "gpt-5.6-luna",
      "api_url": "https://llm-api.amd.com/openai/responses",
      "api_key": "EMPTY",
      "eval_type": "openai_responses_api",
      "generation_config": {
        "reasoning_effort": "medium",
        "max_tokens": 4096,
        "timeout": 7200,
        "extra_query": {
          "api-version": "2025-04-01-preview"
        },
        "extra_headers": {
          "Ocp-Apim-Subscription-Key": "<YOUR_SUBSCRIPTION_KEY>"
        }
      }
    }
  ]
}
```

```bash
evalscope eval \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --datasets aa_lcr \
  --dataset-hub huggingface \
  --eval-batch-size 8 \
  --repeats 16 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":131072,"timeout":7200}' \
  --dataset-args '{"aa_lcr":{"hf_dataset_revision":"9a77ef56b717057ade24ceab4d273712a0b4f19e"}}' \
  --judge "$(cat aa-lcr-judge.json)" \
  --work-dir logs/aa_lcr
```

**IFBench**

```bash
evalscope eval \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --datasets ifbench \
  --dataset-hub huggingface \
  --eval-batch-size 8 \
  --repeats 5 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":131072,"timeout":7200}' \
  --dataset-args '{"ifbench":{"primary_metric":"prompt_level_loose"}}' \
  --work-dir logs/ifbench
```

# License
Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.
