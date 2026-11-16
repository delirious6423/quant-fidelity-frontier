---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3
  - reasoning
  - agentic
  - honesty
pipeline_tag: text-generation
---

![mizar_3.0_constellation](https://cdn-uploads.huggingface.co/production/uploads/692c9b3b93121a11649cec08/nR5pTmCfjbob2SQovxPbm.png)

# Mizar 3.0 27B Abliterated (GGUF)

Mizar 3.0 is a 27B-parameter fine-tune for **honest, persistent agentic reasoning and tool use**. It is the agentic sibling of Alkus 3.1 (pure reasoning); both are trained from an abliterated Qwen3.8-27B base. Mizar's training emphasizes staying honest under tool-use pressure: refusing to fabricate success, hedging claims it cannot verify, and persisting on a hard task instead of declaring a fake "done".

## Details

- **Base:** Qwen3.8-27B (abliterated)
- **Fine-tune:** QLoRA, rank 16 / alpha 32, 1 epoch
- **Format:** GGUF, `Q4_K_M` quantization (~15.8 GB)
- **Context:** up to 131072 tokens
- **Register:** compact, telegraphic reasoning with explicit self-verification

## Recommended runtime settings

- Flash Attention: on
- K/V cache: Q8_0
- Max concurrent requests: 1

## Intended use

Research assistant for multi-step reasoning, code verification, and agentic tool-use workflows where honest "I cannot verify that" behavior is preferred over confident guessing.

## Notes and limitations

This is a research model. It can still produce confident but incorrect details for facts it cannot verify from the context in front of it (for example specific numeric constants, commit hashes, or API names). Verify security-relevant or factual specifics against primary sources. Not intended for production use without human review.
