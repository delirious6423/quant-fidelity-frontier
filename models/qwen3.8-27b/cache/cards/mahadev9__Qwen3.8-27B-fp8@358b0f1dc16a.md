---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - fp8
  - llm-compressor
  - vllm
---

# Qwen3.8-27B-fp8

FP8 (W8A8 dynamic) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor).
`lm_head`, `mlp.down_proj`, `self_attn.o_proj`, and the first/last
transformer layers are kept in the original precision - the layers most
consistently found to be quantization-sensitive - to recover
agentic/tool-calling quality.

## Quality

KL divergence of the quantized model's next-token distribution vs the bf16
baseline (seq len 2048). `prose` is WikiText-2; `agentic` is tool-calling
turns (schema + query + call) rendered with the chat template.

| bucket | tokens | mean KL | top-1 agreement |
| --- | --- | --- | --- |
| prose | 24,130 | 0.0043 | 96.78% |
| agentic | 85,672 | 0.2864 | 90.55% |
| overall | 109,802 | 0.2244 | 91.92% |

## Usage

vLLM:

```bash
vllm serve mahadev9/Qwen3.8-27B-fp8
```

transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mahadev9/Qwen3.8-27B-fp8", torch_dtype="auto", device_map="auto"
)
tok = AutoTokenizer.from_pretrained("mahadev9/Qwen3.8-27B-fp8")
```
