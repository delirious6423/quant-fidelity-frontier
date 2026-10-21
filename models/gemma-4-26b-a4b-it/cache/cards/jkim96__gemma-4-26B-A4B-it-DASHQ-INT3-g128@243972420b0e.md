---
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: transformers
tags:
- dashq
- quantized
- post-training-quantization
- int3
---
![DASH-Q](https://raw.githubusercontent.com/JaeminK/dashq/main/assets/dashq_banner.png)

# gemma-4-26B-A4B-it-DASHQ-INT3-g128

> **DASH-Q** — Diagonal-Aware Shrinkage for Robust PTQ.
> `INT3` · group size 128 · **13.4334 GB** (from 51.6120 GB — **3.8x smaller**)

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer

model = AutoModelForImageTextToText.from_pretrained(
    "jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g128", trust_remote_code=True, device_map="cuda", dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g128")

messages = [{"role": "user", "content": "Explain 2-bit quantization in one sentence."}]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0]))
```

`trust_remote_code=True` is required: the checkpoint ships its quantized-layer
implementation (`modeling_dashq.py`) and Triton kernels (`dashq_kernel.py`).
Without Triton, or on CPU, it falls back to dequantize-and-matmul in PyTorch.

### Requirements

| Package | Minimum | Verified with |
| --- | --- | --- |
| `torch` | 2.4 | 2.12.1+cu130 |
| `transformers` | 5.8 | 5.9.0 |
| `triton` | 3.0 (Linux; bundled with CUDA builds of PyTorch) | 3.7.1 |
| `huggingface_hub` | 1.5 (pulled in by transformers) | 1.15.0 |

## Quantization

| Field | Value |
| --- | --- |
| Base model | `google/gemma-4-26B-A4B-it` |
| Precision | INT3, group size 128 |
| Scale / zero dtype | float16 |
| Calibration | wikitext2, 128 samples x 2048 |
| Size | 13.4334 GB  ·  original 51.6120 GB  ·  3.8x compression |

## Benchmarks

Full zero-shot / few-shot results for every DASH-Q checkpoint:
**[github.com/JaeminK/dashq#benchmarks](https://github.com/JaeminK/dashq#benchmarks)**
