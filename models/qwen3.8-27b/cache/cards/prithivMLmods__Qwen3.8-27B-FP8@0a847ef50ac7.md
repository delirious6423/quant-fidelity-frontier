---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- text-generation-inference
- qwen3.8
- vllm
- fp8
- multimodal
- vision-language-model
- vlm
- image-text-to-text
- image-understanding
- visual-question-answering
- visual-reasoning
- document-understanding
- OCR
- image-captioning
- video-captioning
- video-understanding
- no-mtp
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.8-27B
---

# **Qwen3.8-27B-FP8**

> Qwen3.8-27B-FP8 is a FP8 dynamic quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27-billion-parameter dense causal language model with a native vision encoder from the Qwen team. Built on the Qwen3.5 architectural foundation, it is a compact, deployment-friendly member of the Qwen3.8 generation. This quantization reduces model size and memory footprint while preserving long-form reasoning, mathematical problem solving, scientific analysis, coding, multimodal understanding, and instruction-following capabilities, making deployment more accessible on smaller GPUs. **Qwen3.8-27B** features a 64-layer hybrid architecture that interleaves Gated DeltaNet linear-attention blocks with periodic Gated Attention layers, trained with Multi-Token Prediction. It supports a native 262,144-token context window (extensible to 1M via YaRN scaling), native image and video understanding (from STEM diagrams to hour-scale videos), and flexible thinking control via a `reasoning_effort` parameter (xhigh/medium/low) with thinking enabled by default and historical reasoning preserved across turns. It delivers substantial gains over its predecessor Qwen3.6-27B and is competitive with larger models on agentic coding, computer/browser/mobile-use tasks, and multimodal tool use, while remaining strong on general reasoning benchmarks.

> [!NOTE]
> This model is an experimental release and may generate unexpected behaviors or reasoning artifacts in certain scenarios. Quantization to FP8 may introduce minor numerical differences relative to the bf16 source model.

## Model Size Comparison

| Variant | Approximate Size |
|---|---|
| **Qwen/Qwen3.8-27B** (Full Precision BF16 / FP16) | ~54 GB |
| **prithivMLmods/Qwen3.8-27B-FP8** (Compressed FP8) | ~36 GB |

## Quantization Details

Quantization was performed using [llmcompressor](https://github.com/vllm-project/llm-compressor) with the following recipe:

```yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: ['re:.*lm_head', 're:.*embed_tokens$', 're:.*visual.*', 're:.*model.visual.*',
        're:.*linear_attn.*']
      scheme: FP8_DYNAMIC
      bypass_divisibility_checks: false
      requires_calibration_data: false
```

Linear layers are quantized to FP8 with dynamic per-tensor activation scaling, so no calibration dataset is required (`requires_calibration_data: false`). The `lm_head`, embedding table, any vision-tower (`visual`) components, and `linear_attn` layers are excluded from quantization and remain at full precision to preserve output-head fidelity and numerical stability.

| | |
|---|---|
| **Base model** | Qwen/Qwen3.8-27B |
| **Quantization scheme** | FP8_DYNAMIC (Linear layers only) |
| **Format** | compressed-tensors |
| **Calibration data required** | No (dynamic activation scaling) |
| **Excluded from quantization** | lm_head, embed_tokens, visual (if present), linear_attn |

## Use with vLLM

Qwen3.8-27B-FP8 is served through [vLLM](https://github.com/vllm-project/vllm) with native support for compressed-tensors FP8 checkpoints.

### Requirements

- `torch >= 2.11.0`
- `vllm >=0.27.1
`
- A GPU with FP8 support recommended (Hopper or Blackwell class) for best throughput; also runs on Ampere with FP8 dequantized on the fly.

### Serve

```bash
vllm serve prithivMLmods/Qwen3.8-27B-FP8 \
  --max-model-len 32768
```

### Client request

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

messages = [
    {
        "role": "user",
        "content": "Explain how a transformer model processes text."
    }
]

response = client.chat.completions.create(
    model="prithivMLmods/Qwen3.8-27B-FP8",
    messages=messages,
    temperature=0.0,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

## Quick Start with Transformers

```bash
pip install transformers accelerate
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "prithivMLmods/Qwen3.8-27B-FP8",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "prithivMLmods/Qwen3.8-27B-FP8"
)

messages = [
    {
        "role": "user",
        "content": "Explain how a transformer model processes text."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512
)

print(
    tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )
)
```

## Acknowledgments

- **[llmcompressor](https://github.com/vllm-project/llm-compressor)**: Used to produce the FP8 dynamic quantization for this release.
- **[vLLM](https://github.com/vllm-project/vllm)**: Recommended inference engine with native compressed-tensors FP8 support.