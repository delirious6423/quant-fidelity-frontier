---
license: gemma
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: text-generation
tags:
  - gemma
  - gemma-4
  - mixture-of-experts
  - mxfp4
  - thinking
  - tool-use
  - long-context
---

# Gemma-4-125B-A12B

Gemma-4-125B-A12B is an expanded sparse Mixture-of-Experts language model based
on `google/gemma-4-26B-A4B-it`. This release focuses on agentic coding,
repository understanding, multi-turn tool use, explicit reasoning, and
long-context software tasks.

The model is released in MXFP4 format for the expert weights, with shared and
non-expert weights kept in BF16.

## Model Summary

- Base lineage: `google/gemma-4-26B-A4B-it`
- Architecture: sparse Mixture-of-Experts language model
- Expert layout: `688` total experts
- Active experts per token: `50`
- Total logical text parameters: approximately `125B`
- Active parameter class: approximately `A12B`
- Weight format: MXFP4 experts with BF16 shared weights
- Intended serving mode: Gemma 4 chat, thinking, and tool-use template enabled
- Created on a two-GPU workstation

## Expert Capacity

This checkpoint expands the Gemma 4 expert pool while preserving sparse
inference. Each token activates a selected subset of experts rather than the
full parameter set.

- Expert pool size: `688`
- Active expert budget: `50` experts per token
- Active expert fraction per layer: approximately `7.27%`
- Approximate logical active text size: `11.4B`
- Approximate padded serving active size: `12.2B`

## Recommended Runtime

This model was created on a two-GPU workstation. The following command is the
tested two-GPU serving configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/gemma-4-125b-a12b \
  --served-model-name vllm/doobee \
  --host 0.0.0.0 \
  --port 23333 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 200000 \
  --gpu-memory-utilization 0.96 \
  --trust-remote-code \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --language-model-only \
  --skip-mm-profiling \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --enable-log-requests
```

Use a vLLM build with Gemma 4 MXFP4 MoE support. Long-context serving is
memory-intensive; the command above is configured for a 200k token context on
two high-memory GPUs.

## Chat And Tool Use

The included chat template is intended to be used with thinking enabled.
Tool calling should be exercised through native OpenAI-compatible tool-call
paths rather than raw text parsing.

Recommended evaluation settings:

- Use `temperature=0.0` for deterministic smoke tests.
- Use `temperature=0.2` to `0.7` for normal agentic evaluation.
- Keep thinking enabled for the intended behavior profile.
- Use the included chat template and tokenizer files as shipped.

## Intended Uses

- Agentic coding and software engineering tasks
- Repository exploration and codebase analysis
- Multi-turn tool-use workflows
- Long-context reasoning over technical material
- Patch planning, debugging, and implementation assistance

## Limitations

- This is a large sparse MoE model and requires an inference stack that supports
  Gemma 4 MXFP4 MoE serving.
- The model is optimized for tool-oriented assistant workflows and may not be
  appropriate for all general-purpose chat settings.
- Long-context behavior depends heavily on serving configuration, GPU memory,
  and request batching.

## Credits

Thanks to NVIDIA for providing a broad range of pretraining and post-training
resources that helped make this work possible.
