---
language:
  - en
  - fr
  - multilingual
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - Qwen3.8
  - abliterated
  - NVFP4
  - quantized
  - modelopt
  - reasoning
  - agent
base_model:
  - huihui-ai/Huihui-Qwen3.8-27B-abliterated
  - Qwen/Qwen3.8-27B
---

# Huihui-Qwen3.8-27B-abliterated-NVFP4

NVFP4 (4-bit) quantization of [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated),
produced with NVIDIA TensorRT Model Optimizer 0.45.0 with a **six-source diversity calibration set**
(chat / multilingual / reasoning / agentic tool-call traces / few-shot completion formats).
Ships the fixed chat template from
[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v22.4).

## Read this first: template default and token budget dominate benchmark scores

Every GSM8K number below is the same base model measured under different protocols:

| Protocol (GSM8K) | BF16 base | This NVFP4 |
|---|---|---|
| Raw few-shot completion, 256-token budget | 0.7718 | 0.5982 |
| Chat mode, official template (defaults `xhigh`), 2048-token budget | 0.5900 | — |
| **Chat mode, fixed template (defaults `medium`), 2048-token budget** | **0.9783** | **0.9833** |

- **Real capability is intact** — ~0.98 for both builds; the quantization delta is within noise.
- The official template's `xhigh` default causes overthinking that exhausts any reasonable budget
  (0.59 for the *unquantized* base). The shipped template defaults to `medium`.
- The 4-bit checkpoint drifts away from raw few-shot completion format (spontaneously opens
  `<think>`, then cannot fit its answer in 256 tokens). This held for **four different calibration
  sets** (raw GSM8K 0.598–0.601 in all four) — it is a property of the quantized weights, not of
  calibration. No impact on chat/agent usage; do not feed this model raw few-shot prompts.

## Quantization

- **Format**: NVFP4 weights (group size 16) + FP8 KV cache, ModelOpt 0.45.0 (`hf_ptq.py`), `calib_seq=2048`
- **Calibration (5349 samples, six sources)** — build scripts in `scripts/`:

| Source | Samples | Coverage |
|---|---|---|
| `HuggingFaceH4/ultrachat_200k` | 2048 | English chat |
| `nvidia/Nemotron-SFT-Multilingual-v2` | 2048 | multilingual (hi/ja/ko/pt × code/math/stem) |
| Self-generated long-CoT (`</think>`-terminated, from Vtuber-plan/sharegpt-cleaned prefixes) | 675 | reasoning / thinking termination |
| GSM8K train, raw `Question:/Answer:` format | 300 | plain-completion format |
| [armand0e/claude-fable-5-claude-code](https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code) trajectories | 135 | agentic tool calls (Bash/Edit/WebSearch) |
| [saidutta69/fable-5-premium](https://huggingface.co/datasets/saidutta69/fable-5-premium) (quality ≥ 0.8) | 143 | agentic tool calls / tool results |

- **Excluded modules**: `lm_head`, embeddings, linear-attention `conv1d`/`in_proj_a`/`in_proj_b`, MTP layers, vision tower.
- **Empty-think fix (important)**: both trace datasets ship with thinking stripped (empty
  `<think>` blocks). An intermediate build calibrated on those raw traces regressed badly in chat
  mode (GSM8K 0.6283; ~26% of generations emitted a one-line thinking preamble and then stopped).
  Fix: every assistant turn gets short synthetic `reasoning_content` and windows end only on
  natural boundaries (`scripts/build_v4_traces.py`). With the fix, chat GSM8K returned to 0.9833.

## Validation: xhigh thinking-runaway gate (9 cases)

`gate9_cases.json` (included): FR/EN long-form × temp {0, 0.7} × `reasoning_effort=xhigh`, 32k budget,
8-gram decile copy-rate analysis (separates genuine extended thinking from repetition loops).

| Build | Clean closed stops | Runaway (cap-out / loop-death) |
|---|---|---|
| BF16 base | 7/9 | 2 |
| **This NVFP4 (v4)** | 6/9 | 3 (all at or near temp 0) |
| Existing `qwen3.8-27b` deployment | 2/9 | 7 |

Differences vs the BF16 base are within run-to-run variance; every terminal runaway across every
build occurred at or near temperature 0. **At the default `medium` effort no runaway was ever
observed** in any protocol.

## Known runtime issue (affects all builds of this model family on sglang)

Under **64 concurrent requests × 8192-token generations**, sglang 0.5.17 crashes (0-element reshape
in the linear-attention path, likely a mamba-pool retraction edge case). This is weight-independent
(v2 and v4 both crash). Mitigations: lower `--max-running-requests`, cap per-request `max_tokens`,
or upgrade sglang.

## Usage

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path <this-checkpoint> \
  --quantization modelopt \
  --trust-remote-code \
  --mem-fraction-static 0.8
```

```python
import requests
requests.post("http://127.0.0.1:30000/v1/chat/completions", json={
    "model": "Huihui-Qwen3.8-27B-abliterated-NVFP4",
    "messages": [{"role": "user", "content": "..."}],
    "temperature": 0.7,
    "max_tokens": 8192,
    # only for maximal deliberation (see gate notes):
    # "chat_template_kwargs": {"reasoning_effort": "xhigh"},
})
```

Tool calls: the template renders OpenAI-style `tool_calls`/`tool` history as Qwen `<tool_call>` XML
and tolerates stringified-JSON arguments. History assistant turns render empty `<think>` blocks by
default (canonical Qwen behavior); pass `preserve_thinking=false` to omit them entirely.

> Plain transformers/BF16 inference will not dequantize this checkpoint; use sglang
> `--quantization modelopt` or TensorRT-LLM. 5 shards, each ≤ 5 GB (~19.5 GB total).

## Deployment recommendations

1. Leave `reasoning_effort` at the default (`medium`) — worth +39 GSM8K points over `xhigh` under a
   2048-token budget.
2. `max_tokens` headroom 8k+; the answer follows the closing `</think>` tag.
3. If `xhigh` is required: `temperature > 0`, `repetition_penalty ≈ 1.05`, cap + retry at `medium`.
4. Cap concurrency for long generations (see runtime issue above).
5. Do not use raw few-shot completion prompts.

## License

Apache 2.0 (inherited from the base). Credit to huihui-ai for the abliterated base,
[froggeric](https://huggingface.co/froggeric) for the fixed chat template, and the authors of the
two trajectory datasets used for calibration.
