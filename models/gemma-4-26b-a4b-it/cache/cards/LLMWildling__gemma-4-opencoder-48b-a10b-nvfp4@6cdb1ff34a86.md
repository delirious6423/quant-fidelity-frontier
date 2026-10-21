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
---

# Gemma 4 OpenCoder 48B-A10B

Gemma 4 OpenCoder 48B-A10B is a post-trained Gemma 4 coding model focused on OpenCode-style software engineering workflows, long-context reasoning, and tool use.

The model has approximately **48.1B total text parameters** and approximately **9.5B active text parameters** per token. It supports a **262K token context window** through the underlying Gemma 4 configuration.

This NVFP4 release is intended for efficient serving in runtimes with Gemma 4 NVFP4, reasoning, and tool-calling support.

## Intended use

- OpenCode coding workflows
- Agentic software engineering tasks
- Long-context codebase navigation
- Reasoning-enabled tool use
- Multi-file implementation and editing sessions

Run with reasoning and tool calling enabled.

## Format

- Format: NVFP4 / compressed-tensors
- Architecture: Gemma 4
- Total text parameters: ~48.1B
- Active text parameters: ~9.5B
- Context length: 262,144 tokens
- Language focus: English

## Example outputs

In one continuous OpenCode chat, this checkpoint one-shot generated:

- a Flappy Bird-style game
- a Breakout game
- a skiing game
- a monster truck jumping over cars

![Flappy Bird demo](images/flappy-bird.png)

![Breakout demo](images/breakout.png)

![Ski game demo](images/ski.png)

![Monster truck demo](images/monster-truck.png)

## Serving

Use a Gemma 4 compatible runtime with reasoning and tool calling enabled. For NVFP4 serving, use a runtime build that supports Gemma 4 compressed-tensors NVFP4 MoE kernels.

Example:

```bash
vllm serve <repo-or-local-path> \
  --trust-remote-code \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice
```

For best results in OpenCode, keep the model's native chat template and enable thinking/reasoning mode.
