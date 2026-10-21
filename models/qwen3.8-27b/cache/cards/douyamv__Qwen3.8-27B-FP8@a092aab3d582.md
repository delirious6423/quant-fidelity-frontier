---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- qwen
- qwen3
- fp8
- quantized
- vllm
- safetensors
language:
- en
- zh
---

# Qwen3.8-27B-FP8

<p>
  <a href="https://huggingface.co/Qwen/Qwen3.8-27B"><img src="https://img.shields.io/badge/Base_Model-Qwen3.8--27B-blue" alt="Base Model"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
  <img src="https://img.shields.io/badge/Quantization-FP8_Dynamic-orange" alt="FP8">
</p>

FP8 dynamically quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in safetensors format for efficient inference with reduced VRAM usage.

## Model Details

| Property | Value |
|---|---|
| Base Model | Qwen/Qwen3.8-27B |
| Quantization | FP8 Dynamic (W8A8) |
| Format | Safetensors (2 shards) |
| Total Size | ~27 GB |
| VRAM Reduction | ~50% vs BF16 |
| Quality Loss | Minimal |

## Overview

This model applies FP8 dynamic quantization to all linear layers, reducing model size by approximately 50% while maintaining near-original accuracy. FP8 quantization provides the best quality-to-size ratio among quantization methods.

## Usage

### vLLM

```bash
vllm serve douyamv/Qwen3.8-27B-FP8 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --trust-remote-code
```

### SGLang

```bash
python -m sglang.launch_server \
    --model-path douyamv/Qwen3.8-27B-FP8 \
    --tp 1 \
    --trust-remote-code
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "douyamv/Qwen3.8-27B-FP8",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("douyamv/Qwen3.8-27B-FP8")

messages = [{"role": "user", "content": "Hello, tell me about yourself"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Base Model Information

- **Model:** Qwen3.8-27B
- **Parameters:** 27.78B
- **Architecture:** Hybrid (Gated DeltaNet + Gated Attention)
- **Context Length:** 262,144 tokens (extensible to 1M+)
- **License:** Apache 2.0
- **Capabilities:** Text generation, coding, reasoning, vision understanding, tool calling

## Quantization Method

Quantized using [llm-compressor](https://github.com/vllm-project/llm-compressor) with `FP8_DYNAMIC` scheme applied to all `Linear` layers.

## Credits

- Original model by [Qwen Team](https://huggingface.co/Qwen)
- FP8 quantization by [douyamv](https://huggingface.co/douyamv)

## Related Models

- [douyamv/Qwen3.8-27B-GGUF](https://huggingface.co/douyamv/Qwen3.8-27B-GGUF) — GGUF quantizations (Q2_K to Q8_0)
- [douyamv/Qwen3.8-27B-abliterated](https://huggingface.co/douyamv/Qwen3.8-27B-abliterated) — Abliterated (uncensored) version
- [douyamv/Qwen3.8-27B-abliterated-GGUF](https://huggingface.co/douyamv/Qwen3.8-27B-abliterated-GGUF) — Abliterated GGUF quantizations
