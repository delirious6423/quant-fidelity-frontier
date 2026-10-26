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

The model was evaluated on the GSM8K benchmark using the [vLLM](https://docs.vllm.ai/en/latest/) framework
with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). 

### Accuracy

| **Benchmark**                     | Qwen/Qwen3.8-Flash-Next | amd/Qwen3.8-Flash-Next-Quark-MXFP4 (this model) | **Recovery** |
| --------------------------------- | --------------------------------- | ---------------------- | ------------ |
| gsm8k (flexible-extract, 5-shot)  | 96.8                         | 96.5              | 99.7%    |

### Reproduction

The GSM8K results were obtained using the vLLM framework, based on the Docker image `vllm/vllm-openai-rocm:v0.32.0`, with lm-eval installed from source. This Reproduction serves the model without MTP. Enabling MTP on the formal `v0.32.0` release currently requires [vLLM PR #61770](https://github.com/vllm-project/vllm/pull/61770).

1. Serve the model:
```
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
  vllm serve amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --tensor-parallel-size 2 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --trust-remote-code  \
  --port <port>

```

2. Run lm-evaluation-harness against the running server (chat-completions endpoint, chat template
applied, 5-shot, greedy):
```
lm_eval --model local-chat-completions \
    --model_args "model=amd/Qwen3.8-Flash-Next-Quark-MXFP4,base_url=http://localhost:<port>/v1/chat/completions,num_concurrent=32,timeout=1200,max_retries=3,tokenized_requests=False,max_gen_toks=2048" \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template
```

# License
Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.
