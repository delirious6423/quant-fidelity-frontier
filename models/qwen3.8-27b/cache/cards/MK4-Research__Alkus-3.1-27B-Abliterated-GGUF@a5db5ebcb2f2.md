---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3
  - reasoning
  - honesty
pipeline_tag: text-generation
---

![alkus_3.1_constellation](https://cdn-uploads.huggingface.co/production/uploads/692c9b3b93121a11649cec08/RGNh5KrWxN8kztzwzpD6m.png)

# Alkus 3.1 27B Abliterated (GGUF)

Alkus 3.1 is a 27B-parameter fine-tune for **compact, honest, persistent reasoning**. It is the pure-reasoning sibling of Mizar 3.0 (agentic tool use); both are trained from an abliterated Qwen3.8-27B base. Alkus favors careful step-by-step reasoning with explicit self-verification, and is tuned to self-correct and to prefer saying what it does not know over fabricating an answer.

## Details

- **Base:** Qwen3.8-27B (abliterated)
- **Fine-tune:** QLoRA, rank 16 / alpha 32, 2 epochs
- **Format:** GGUF, `Q4_K_M` quantization (~15.8 GB)
- **Context:** up to 131072 tokens
- **Register:** compact, telegraphic reasoning with explicit self-verification

## Recommended runtime settings

- Flash Attention: on
- K/V cache: Q8_0
- Max concurrent requests: 1

## Intended use

Research assistant for multi-step reasoning and problem solving where a compact, self-checking chain of thought and honest uncertainty are preferred over confident guessing.

## Notes and limitations

This is a research model. It can still produce confident but incorrect details for facts it cannot verify from the context in front of it (for example specific numeric constants, commit hashes, or API names). Verify security-relevant or factual specifics against primary sources. Not intended for production use without human review.
