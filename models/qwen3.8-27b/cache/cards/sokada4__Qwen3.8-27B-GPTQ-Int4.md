---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
  - gptq
  - 4bit
  - quantized
  - qwen3.8
  - vllm
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B — GPTQ 4-bit (W4A16)

4-bit GPTQ quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [GPTQModel](https://github.com/ModelCloud/GPTQModel).

## Quantization details

| Setting | Value |
|---|---|
| Method | GPTQ, W4A16 |
| Bits | 4 |
| Group size | 128 |
| Symmetric | yes |
| `desc_act` | false |
| Quantizer | GPTQModel 7.3.4 |
| Calibration data | `allenai/c4` (zh/en/ja) + `codeparrot/codeparrot-clean-valid`, 256 samples total (64 per source), seq_len 4096 |

The `mtp.*` tensors (the model's Multi-Token-Prediction / speculative-decoding draft head, 15 tensors) are **kept in bf16, not quantized** — GPTQModel's `qwen3_5.py` model definition preserves them via its `out_of_model_tensors` mechanism rather than passing them through the quantization loop at all (upstream `transformers` doesn't load MTP weights into the normal inference graph in the first place, so there is nothing to quantize there). This means the MTP speculative-decoding path below works out of the box.

Vision tower weights are similarly untouched — only the `language_model.*` weights are quantized (this checkpoint is intended for **text-only serving**).

## Serving with vLLM

Tested with vLLM 0.26.0. A dense 27B model at 4-bit needs ~17.7GiB just for weights, which is tight on a single 24GB consumer GPU once you add KV cache — 2x24GB via tensor parallelism gives comfortable headroom, including for MTP speculative decoding:

```bash
vllm serve <this-repo> \
  --served-model-name qwen3.8-27b-gptq \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 200000 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

Notes:
- `--tool-call-parser qwen3_xml` and `qwen3_coder` are aliases for the identical parser class in vLLM — either works.
- `--max-model-len 200000` (vs. the model's native 262144) is needed to leave enough KV-cache headroom for the MTP draft path's own verification buffers on 2x24GB. If you have more VRAM (e.g. 4x24GB via `--tensor-parallel-size 4`, or 2x48GB+ cards) you likely won't need this cut.
- Single-GPU (24GB) serving is only practical without MTP speculative decoding, and even then needs `--max-model-len` reduced well below native (a single 24GB card OOMs on the CUDA-graph-capture step at native length regardless of KV-cache tuning).

### Benchmark (`vllm bench serve`, random dataset, input-len 128 / output-len 64, TP=2, 2x RTX 3090, MTP speculative decoding enabled)

| concurrency | mean TTFT | mean TPOT | output tok/s | MTP acceptance rate |
|---|---|---|---|---|
| 1 | 198.5ms | 12.6ms | 64.6 | 74.8% |
| 8 | 488.0ms | 43.9ms | 154.3 | 74.7% |

For reference, the sibling [Qwen3.6-27B GPTQ checkpoint](https://huggingface.co/sokada4/Qwen3.6-27B-GPTQ-Int4) — architecturally identical, same quantization recipe, same hardware — measured a noticeably higher MTP acceptance rate (84-88%). Raw decode speed (TTFT/TPOT) between the two models is close, as expected since they share the same architecture, but acceptance rate is a property of *this specific model's* co-trained MTP head, not the architecture — measure it yourself if you're relying on speculative decoding for latency.

## Using as a Claude Code / Codex CLI backend

Both point their OpenAI/Anthropic-compatible client at your vLLM server.

**Claude Code** (Anthropic Messages API, vLLM implements this natively):

```bash
export ANTHROPIC_BASE_URL="http://<your-vllm-host>:8000"
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_AUTH_TOKEN="dummy"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3.8-27b-gptq"
export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3.8-27b-gptq"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3.8-27b-gptq"
# vLLM's prefix caching is broken by Claude Code's per-request attribution header:
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
# This model's vLLM server only accepts reasoning_effort in {xhigh, medium, low}
# (not Claude Code's default "high") — override explicitly:
claude --effort xhigh
```

**Codex CLI** (OpenAI-compatible):

```bash
codex \
  -c preferred_auth_method="apikey" \
  -c model="qwen3.8-27b-gptq" \
  -c model_provider="local-vllm" \
  -c model_providers.local-vllm.name="local-vllm" \
  -c model_providers.local-vllm.base_url="http://<your-vllm-host>:8000/v1" \
  -c model_providers.local-vllm.wire_api="responses" \
  -c model_providers.local-vllm.env_key="DUMMY_KEY" \
  -c model_reasoning_effort="xhigh"
```

Note the `xhigh` reasoning effort above — unlike our Qwen3.6-27B checkpoint (which accepts `high`), this model's chat template/server only accepts `{xhigh, medium, low}`; passing `high` returns a 400 error.

**Tool-call reliability caveat:** in independent field testing, local Qwen3.6/3.8-family deployments have shown a meaningfully higher malformed-tool-call rate than Claude/GPT-class hosted models (community reports put it around 10x higher for Qwen3.6). Gate tool calls / keep a human in the loop rather than running fully autonomous agent loops against this checkpoint.
