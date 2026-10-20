---
license: gemma
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - gemma4
  - code
  - opencode
  - reasoning
  - tool-calling
  - nvfp4
  - compressed-tensors
---

# Gemma 4 OpenCoder 40B-A8B NVFP4 v2

Gemma 4 OpenCoder 40B-A8B v2 is a continued post-trained Gemma 4 coding
model focused on agentic software engineering, long-context reasoning, and
reliable tool use. It is designed for OpenCode workflows including repository
navigation, implementation, editing, debugging, review, and sustained
multi-file work.

The checkpoint contains approximately **42.4B text parameters** and activates
approximately **8.1B text parameters** per token. It retains Gemma 4's native
**262,144-token context window**.

This release is a self-contained NVFP4 / compressed-tensors checkpoint using
the standard Gemma 4 architecture. It does not require an adapter or auxiliary
model.

## Intended use

- Agentic software engineering
- OpenCode coding workflows
- Long-context codebase navigation
- Reasoning-enabled tool use
- Multi-file implementation and editing
- Debugging and code review

Run with reasoning and automatic tool calling enabled, and preserve the
checkpoint's included chat template.

## Model format

- Weight format: NVFP4 / compressed-tensors
- Architecture: Gemma 4 Mixture-of-Experts
- Total text parameters: approximately 42.4B
- Active text parameters: approximately 8.1B per token
- Maximum context: 262,144 tokens

## vLLM

Use a recent vLLM release with Gemma 4, compressed-tensors NVFP4, reasoning,
and tool-parser support.

```bash
vllm serve LLMWildling/gemma-4-opencoder-40b-a8b-nvfp4-v2 \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --max-model-len 262144
```

For OpenCode, select the served model and keep reasoning and tool calling
enabled.
