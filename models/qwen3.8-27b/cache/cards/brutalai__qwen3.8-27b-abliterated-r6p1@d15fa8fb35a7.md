---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
  - en
library_name: vllm
pipeline_tag: text-generation
tags:
  - abliterated
  - uncensored
  - qwen3
---

# Qwen3.8-27B · Brutal AI Abliterated

A refusal-reduced derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced by **Brutal AI**.

## Specification

| | |
|---|---|
| Base model | Qwen3.8-27B (Apache-2.0) |
| Architecture | 64 layers, dense |
| Format | FP8, E4M3, 128×128 block scales |
| Size | ~29 GB, 25 shards |
| Context | 32,768 tokens |
| Serving | vLLM, tensor-parallel 1 on a single H200 |
| Sampling | standard, no system prompt or flags required |

The modification is a targeted weight edit applied to a subset of attention projections, reducing
the model's tendency to refuse. It is a weight edit, not a fine-tune, adapter or prompt wrapper.

## Usage

```sh
vllm serve brutalai/qwen3.8-27b-abliterated-r6p1 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --max-num-seqs 64 --gpu-memory-utilization 0.90
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="brutalai/qwen3.8-27b-abliterated-r6p1",
    messages=[{"role": "user", "content": "Explain options pricing in one paragraph."}],
    max_tokens=512,
)
print(r.choices[0].message.content)
```

## Benchmarks

Identical tasks, unmodified FP8 base vs this model, one run per arm.

| Measure | Base | This model |
|---|---:|---:|
| Harmful prompts answered (of 100) | 4 | **100** |
| Benign prompts falsely refused (of 250) | 16 | **2** |
| LiveCodeBench pass@1 (of 175) | 92 | 91 |
| IFBench loose / strict (of 300) | 116 / 108 | **120 / 110** |
| MMLU-Pro (of 140) | 102 | 99 |
| FinQA (of 100) | 46 | **49** |

## Limitations

- Not safety-tuned: removing refusals removes a guardrail. Review outputs before shipping.
- Can produce longer responses than the base model; set an output cap.
- Inherits the factual limits, bias and failure modes of the base model.
- Evaluated in English, single-turn, thinking disabled, held-out split, single run per arm.

## About Brutal AI

Brutal AI is a company pioneering AI-driven software development.

- Website: https://brutal.ai/en
- Models: https://huggingface.co/brutalai

## Licence

Derived from Qwen3.8-27B and distributed under **Apache-2.0**, following the base model's licence.
Qwen and Qwen3 are trademarks of their respective owners; this derivative is not endorsed by them.
