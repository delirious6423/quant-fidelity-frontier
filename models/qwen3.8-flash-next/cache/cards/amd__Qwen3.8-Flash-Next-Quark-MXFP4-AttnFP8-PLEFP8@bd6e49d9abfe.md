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
- **Supported Hardware Microarchitecture:** AMD MI350X
- **ROCm**: 7.2.4
- **PyTorch**: 2.11.0+rocm7.2
- **Transformers**: 5.12.1
- **Operating System(s):** Linux
- **Inference Engine:** [SGLang](https://docs.sglang.ai/) (PR #36601 and PR #39064)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html) (v0.13 with Patch for PLE Quantization)
  - **Quantized layers:** All MoE experts in the language model, **including the shared expert and the MTP experts**, plus the full self-attention and linear-attention projections, including MTP self-attention. PLE embedding table is quantized to Per-Tensor FP8.
- **Weight quantization:**
  - self_attn and linear_attn Perchannel, FP8E4M3, Static
  - MOE OCP MXFP4, Static
  - PLE embedding table Per-Tensor, FP8E4M3, Static
- **Activation quantization:**
  - self_attn and linear_attn Pertoken, FP8E4M3, Dynamic
  - MOE OCP MXFP4, Dynamic

This checkpoint quantizes the routed **and** shared MoE experts to OCP MXFP4, including the MTP experts. Self-attention and linear-attention projections are quantized to FP8, including MTP self-attention. The PLE embedding table is quantized to Per-Tensor FP8. Everything else is kept in bf16, including linear-attention `conv1d`, the PLE conv/proj/norm path, `mtp.fc_embedding`, `mtp.fc_hidden`, router gates, the shared-expert gate, the vision tower, hyper-connection mixers, and `lm_head`.

# Model Quantization

The model was quantized from [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) using [AMD-Quark](https://quark.docs.amd.com/latest/index.html). MoE expert weights and activations are quantized to OCP MXFP4, self-attention and linear-attention are quantized to FP8, and the PLE embedding table is quantized to Per-Tensor FP8.

**Quantization script:**

```python
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.quantization.config.type import ScaleType

SRC = "Qwen/Qwen3.8-Flash-Next"
OUTPUT = "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
NGRAM = "*ple.ple_embedding.ngram_embedding*"
EXCLUDE = [
    "lm_head", "model.visual.*",
    "*mlp.gate", "*shared_expert_gate*", "*hyper_connection*",
    "*linear_attn.conv1d",
    "*ple.conv1d", "*ple.key_proj", "*ple.value_proj",
    "*ple.norm_conv", "*ple.norm_key", "*ple.norm_query",
    "mtp.fc_embedding", "mtp.fc_hidden",
]

template = LLMTemplate.get("qwen4_exp")
quant_config = template.get_config(
    scheme="mxfp4",
    exclude_layers=EXCLUDE,
    layer_config={
        NGRAM: "fp8",
        "*linear_attn.*": "ptpc_fp8",
        "*self_attn.*": "ptpc_fp8",
    },
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

This checkpoint is validated for native MXFP4 and FP8 deployment on AMD Instinct MI350X GPUs. The accuracy measurements in the below section were collected on MI350X with the following environment setup:

- SGLang from [PR #36601](https://github.com/sgl-project/sglang/pull/36601) and [PR #39064](https://github.com/sgl-project/sglang/pull/39064).
- Container image `rocm/sgl-dev:v0.5.19-rocm724-mi35x-20260916` with the above patches applied.

### Accuracy

The model was evaluated on the GSM8K benchmark using the [SGLang](https://docs.sglang.ai/) framework with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The baseline is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Both sides use MTP. The scores are the median of five runs.

| **Benchmark** | Qwen/Qwen3.8-Flash-Next | amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 (this model) | **Recovery** |
| --- | --- | --- | --- |
| gsm8k (flexible-extract, 5-shot) | 96.89 | 96.21 | 99.3% |

### Reproduction

The GSM8K results were obtained using the SGLang framework, based on the Docker image `rocm/sgl-dev:v0.5.19-rocm724-mi35x-20260916`, with the diffs from [PR #36601](https://github.com/sgl-project/sglang/pull/36601) and [PR #39064](https://github.com/sgl-project/sglang/pull/39064) applied inside the container, and lm-eval installed from source.

1. Serve the model. For the BF16 baseline, change `--model-path` to `Qwen/Qwen3.8-Flash-Next`.

```bash
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_USE_AITER_FP8_PER_TOKEN=1

sglang serve --model-path amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 \
    --tp 2 \
    --attention-backend aiter \
    --trust-remote-code \
    --mem-fraction-static 0.9 \
    --chunked-prefill-size -1 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

2. Run lm-evaluation-harness against the running server (chat-completions endpoint, chat template applied, 5-shot, greedy):

```bash
lm_eval --model local-chat-completions \
    --model_args "model=amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=32,timeout=1200,max_retries=3,tokenized_requests=False,max_gen_toks=2048" \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template \
    --gen_kwargs '{"chat_template_kwargs":{"enable_thinking":false}}'
```

# License

Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.