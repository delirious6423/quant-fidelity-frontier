---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: vllm
language: [en, ko, zh, ja]
tags: [fp8, quantized, vidraft, long-context, reasoning]
---

# Qwen3.8-27B-FP8

FP8 quantization of **Qwen3.8-27B**, prepared and served by **VIDRAFT**.

Original weights, unmodified in behaviour. Quantized to FP8 for efficient
high-throughput serving with the full **262,144-token** context window intact.

| | |
|---|---|
| Base model | Qwen/Qwen3.8-27B |
| Weights | FP8 |
| KV cache | FP8 |
| Context | 262,144 tokens |
| Parameters | 27B (64 layers, hidden 5120, 24 heads / 4 KV heads) |
| Languages | English, Korean, Chinese, Japanese, and more |
| License | Apache-2.0 (inherited) |

---

## Why serve this from VIDRAFT

We are the team that took **#1 on the Google x Hugging Face "The Fast Gemma Challenge"**
leaderboard as of **August 2, 2026**, with a **verified** record of **510.58 tok/s**.

What that number means:

- The challenge's own baseline - Google's official QAT W4A16 build, measured on the
  same harness and the same hardware - runs at **95.36 tok/s**.
  Our entry reached **510.58 tok/s**: a **5.35x** speed-up.
- It came from **serving optimization alone**. No weight retraining, no distillation
  of the target model, no custom silicon. Same checkpoint, same GPU.
- The challenge re-measures every submission and marks it `invalid` when the result
  does not reproduce. Of **718** submissions, only **40** were verified.
  Ours is one of them.

That is the engineering we apply to this endpoint.

> Leaderboard is public and continuously updated; rankings move as new entries are
> verified. The figures above are the verified record as of the date stated.

---

## Serving characteristics

- **Full 262K context.** Not a truncated or sliding-window approximation - the model
  is served at its native maximum.
- **Prompt caching.** Repeated prefixes (system prompts, tool definitions, long
  documents) are billed at a reduced rate and returned faster. Cached token counts
  are reported in the standard `usage` object, so cost is auditable per request.
- **Reasoning content is separated.** The model's chain of thought is returned in a
  dedicated field, not mixed into the answer. Clients that ignore the field get a
  clean response; clients that want it can display it.
- **OpenAI-compatible API,** including tool calling and streaming.

---

## Usage

```python
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="...")

resp = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[{"role": "user", "content": "Summarise this contract."}],
)
print(resp.choices[0].message.content)
```

---

## About VIDRAFT

VIDRAFT is a Korean AI company building sovereign foundation models and inference
systems. Selected public results:

- **Fast Gemma Challenge** - #1 verified record as of 2026-08-02 (510.58 tok/s)
- **AETHER** - a from-scratch open foundation model family, with training data,
  code, and logs released
- **Darwin** - a reasoning model family
- Research platforms for drug and materials discovery

Links: [vidraft.net](https://vidraft.net) | [Hugging Face](https://huggingface.co/VIDraft)

---

## Notes

- This repository contains quantized weights and configuration only.
- Behaviour follows the base model. Refer to Qwen's model card for capabilities,
  evaluation results, and intended use.
- Apache-2.0, inherited from the base model.
