---
license: mit
language:
  - en
tags:
  - deepseek
  - moe
  - sovereign
  - text-generation
  - agentic
  - speculative-decoding
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
inference: false
---

# 🌌 deepseek-v4-sovereign — the new sovereign AI model

**SOVEREIGN's flagship model.** A sovereign-tuned derivative of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(304B MoE, MIT license) with an extended reasoning budget, 1M-token context,
and DSpark speculative decoding — tuned on private, user-owned corpora only.

## Model card

| Field | Value |
|---|---|
| Base | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Architecture | Mixture-of-Experts (MoE) + speculative decoding (DSpark) |
| Parameters | 304B total, fraction activated per token |
| Context window | 1,048,576 tokens (1M) |
| Max output | 384K tokens (high/max reasoning) |
| Reasoning effort | `low` / `high` / `max` |
| Precision | BF16 / FP16 / FP32 / FP8 (E4M3, E2M1) / INT8 |
| Quantizations | 90+ model tree (GGUF/safetensors) |
| License | MIT (derived) |
| Paper | arXiv:2606.19348 |
| Sovereign property | weights + fine-tunes stored locally; zero mandatory telemetry |

## Why "sovereign"

- Runs fully locally (vLLM / SGLang / transformers) or via your own VPC.
- No mandatory external API calls; HF router (`router.huggingface.co/v1`)
  is an *option*, not a dependency.
- Fine-tuning data, adapters, and inference logs stay under your control.

## Deployment

```bash
# vLLM (4×GB300 node) — DSpark speculative decoding enabled with one flag
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'

# SGLang
sglang serve --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 4 --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK --chunked-prefill-size 4096
```

Recommended sampling for agentic scenarios: `temperature=1.0, top_p=0.95`;
otherwise `top_p=1.0`.

## Chat template

No Jinja template ships with this release — encode via the OpenAI-compatible
contract: `encoding_dsv4.encode_messages(messages, thinking_mode="thinking",
reasoning_effort="max")` (see `models/deepseek-v4-flash-0731/` docs and
`api_examples/`).
