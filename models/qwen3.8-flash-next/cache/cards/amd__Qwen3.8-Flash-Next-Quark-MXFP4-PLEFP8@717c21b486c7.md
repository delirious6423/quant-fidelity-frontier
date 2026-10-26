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
- **ROCm**: 7.2.4
- **PyTorch**: 2.11.0+rocm7.2
- **Transformers**: 5.12.1
- **Operating System(s):** Linux
- **Inference Engine:** [SGLang](https://docs.sglang.ai/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html)
  - **Quantized layers:** All MoE experts in the language model, **including the shared expert**. PLE embedding table is quantized to Per-Tensor FP8. Attention is kept in bf16.
  - **Weight quantization:**
    - MOE OCP MXFP4 for MoE experts, Per-Tensor FP8 for PLE embedding table, Static
  - **Activation quantization:**
    - MOE OCP MXFP4 for MoE experts, Dynamic

This checkpoint quantizes the routed **and** shared MoE experts to OCP MXFP4, PLE embedding table is quantized 
to Per-Tensor FP8, everything else is kept in bf16, including the full self-attention and the 
(Mamba-style) linear-attention projections, the MTP head, router gates, shared-expert gate, vision tower,
hyper-connection mixers, and `lm_head`.

# Model Quantization

The model was quantized from [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) using [AMD-Quark](https://quark.docs.amd.com/latest/index.html). MoE expert weights and
activations are quantized to OCP MXFP4, PLE embedding table is quantized to Per-Tensor FP8. 

**Quantization script:**

```python
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.quantization.config.type import ScaleType

SRC = "Qwen/Qwen3.8-Flash-Next"
OUTPUT = "amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8"
NGRAM = "*ple.ple_embedding.ngram_embedding*"
EXCLUDE = [
    "lm_head", "model.visual.*", "mtp.*",
    "*mlp.gate", "*shared_expert_gate*",
    "*.linear_attn.*", "*.self_attn.*", "*hyper_connection*",
    "*ple.conv1d", "*ple.key_proj", "*ple.value_proj",
    "*ple.norm_conv", "*ple.norm_key", "*ple.norm_query",
]

template = LLMTemplate.get("qwen4_exp")
quant_config = template.get_config(
    scheme="mxfp4",
    exclude_layers=EXCLUDE,
    layer_config={NGRAM: "fp8"},
    weight_scale_groups=["*ple.ple_embedding.ngram_embedding.shard_*.weight"],
)
quant_config.layer_quant_config[NGRAM].weight.scale_type = ScaleType.bfloat16

ModelQuantizer(quant_config).direct_quantize_checkpoint(
    pretrained_model_path=SRC,
    save_path=OUTPUT,
    weight_converters=template.f2f_weight_converters,
    device="cuda",
)

```

For further details or issues, please refer to the AMD-Quark documentation.

# Deployment and Evaluation

### Use with SGLang

This checkpoint is validated for native MXFP4 deployment on AMD Instinct MI350X and MI355X GPUs. 
The accuracy measurements in the below section were collected on MI355X with the following environemnt setup:
- Container image `lmsysorg/sglang:v0.5.22-rocm10-mi35x` ([SGLang v0.5.22](https://github.com/sgl-project/sglang/releases/tags/v0.5.22)).

### Accuracy

The model was evaluated on the GSM8K benchmark using the [SGLang](https://docs.sglang.ai/) framework
with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). 

| **Benchmark**                     | Qwen/Qwen3.8-Flash-Next | amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8 (this model) | **Recovery** |
| --------------------------------- | --------------------------------- | ---------------------- | ------------ |
| gsm8k (flexible-extract, 5-shot)  | 96.97                         | 96.44              | 99.5%    |

### Reproduction

The GSM8K results were obtained using the SGLang framework, based on the Docker image `lmsysorg/sglang:v0.5.22-rocm10-mi35x`, with lm-eval installed from source.
1. Serve the model:
```
sglang serve --model-path amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8 \
    --tp 4 \
    --attention-backend triton
```

2. Run lm-evaluation-harness against the running server (chat-completions endpoint, chat template
applied, 5-shot, greedy):
```
lm_eval --model local-chat-completions \
    --model_args "model=amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=32,timeout=1200,max_retries=3,tokenized_requests=False,max_gen_toks=2048" \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template --gen_kwargs '{"chat_template_kwargs":{"enable_thinking":false}}'
```

# License
Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.

