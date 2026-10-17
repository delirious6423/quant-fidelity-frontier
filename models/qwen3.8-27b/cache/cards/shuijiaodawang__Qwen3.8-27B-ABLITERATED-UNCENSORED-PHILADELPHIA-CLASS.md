---
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
language:
  - multilingual
tags:
  - qwen
  - qwen3.8
  - qwen3_5
  - transformers
  - safetensors
  - gguf
  - bfloat16
  - multimodal
  - image-text-to-text
  - text-generation
  - conversational
  - abliterated
  - uncensored
  - refusal-reduction
---

# Qwen3.8 27B - ABLITERATED UNCENSORED PHILADELPHIA CLASS

**0 refusals across an internal 842-prompt screen. 0 refusals across a separate 126-prompt family holdout. 23/24 coherence checks passed.**

PHILADELPHIA CLASS is a BF16 multimodal derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), modified to sharply reduce refusal behavior while retaining the upstream hybrid text-and-vision backbone.

## Results

| Internal evaluation | Result |
|---|---:|
| Full refusal screen - 842 prompts | **0/842 refusals; 100% usable; 0 degeneration** |
| Family-disjoint holdout - 126 prompts, 96 generated tokens | **0/126 refusals; 100% usable; 0 degeneration** |
| Coherence regression - coding, JSON, debugging, explanation, math, and boundary tasks | **23/24 passed** |
| Long-form diagnostic - 24 prompts, 256 generated tokens | **0/24 refusals; 23/24 usable; 0 degeneration** |
| Fresh multimodal reload smoke | **Passed; correctly identified a blue square** |

These are automated internal development evaluations, not public leaderboards or independent audits. The 842-prompt screen includes the 716 prompts used to fit the transformation; the separate 126-prompt result uses held-out prompt families. "Usable" measures response form and topicality, not factual accuracy. Results above were measured on the BF16 checkpoint with thinking disabled. Quantization and backend changes can affect behavior.

## Files

| File | Use |
|---|---|
| `model.safetensors` | Single-file BF16 Transformers checkpoint with text and vision weights |
| `Q4_K_M-Qwen3.8-27B-ABLITERATED-UNCENSORED-PHILADELPHIA-CLASS.gguf` | Smaller local text-generation GGUF |
| `Q5_K_M-Qwen3.8-27B-ABLITERATED-UNCENSORED-PHILADELPHIA-CLASS.gguf` | Balanced local text-generation GGUF |
| `Q8_0-Qwen3.8-27B-ABLITERATED-UNCENSORED-PHILADELPHIA-CLASS.gguf` | Higher-precision local text-generation GGUF |

Use the Safetensors checkpoint for the validated multimodal path. The GGUF files are text-only unless a matching vision projector is explicitly provided.

## Transformers

```bash
pip install -U "transformers>=5.14.1" accelerate safetensors
```

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "KridgeDookie/Qwen3.8-27B-ABLITERATED-UNCENSORED-PHILADELPHIA-CLASS"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Explain why the sky appears blue."}],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to(model.device)

input_length = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

print(processor.batch_decode(
    output[:, input_length:],
    skip_special_tokens=True,
)[0])
```

Qwen3.8 thinking mode remains available by omitting `enable_thinking=False` or setting it to `True`. Plan for roughly 56 GB for the BF16 weights, plus runtime and KV-cache overhead.

## Ollama

Download a GGUF and place this `Modelfile` beside it:

```text
FROM ./Q4_K_M-Qwen3.8-27B-ABLITERATED-UNCENSORED-PHILADELPHIA-CLASS.gguf
PARAMETER num_ctx 32768
```

```bash
ollama create qwen3.8-27b-philadelphia-class:q4_k_m -f Modelfile
ollama run qwen3.8-27b-philadelphia-class:q4_k_m
```

## Notes

- "Uncensored" describes strong refusal reduction; it is not a guarantee for every prompt, language, decoding configuration, quantization, or runtime.
- The upstream MTP head is not included. Standard generation and thinking remain available, but MTP-dependent speculative decoding is not supported.
- This release does not claim that upstream reasoning, factuality, coding, or vision benchmark scores were preserved unchanged.

## Attribution

Derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
