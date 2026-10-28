---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
tags:
  - cyberneurova
  - uncensored
  - abliterated
  - deepseek
  - moe
  - fp8
  - reasoning
---

# cyberneurova-DeepSeek-V4.1-Flash-1017

An uncensored, neutral assistant built by **CyberNeurova** on top of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
It answers directly and does not add unsolicited disclaimers, warnings, or caveats.
Whatever tone or policy you want is set entirely by your own system prompt.

💬 **Chat with it:** [cyberneurova.ai](https://cyberneurova.ai)
✉️ **Support:** support@cyberneurova.com

---

## Highlights

- **Massive MoE** — 763B total parameters, ~37B active per token. The full weight
  of the model is present; only the experts that match the token route.
- **1M native context** (1,048,576 tokens), multilingual (English + Chinese), tool-calling.
- **Reasoning model** — supports an explicit thinking trace via `thinking` mode, and
  is uncensored in **both** modes: with thinking on *and* with thinking off.
- Direct by default; fully steerable — send a system prompt and it is honored
  verbatim, with no default behavior merged in.
- **Capability preserved.** Coding, reasoning, and general knowledge are intact;
  only the refusal behavior is removed.
- **FP8 checkpoint** — ships in the base model's native FP8 precision
  (float8_e4m3fn), same size and serving speed as the original.

---

## Usage

DeepSeek-V4.1-Flash uses a new MoE architecture with Hyper-Connections and
384 routed experts per layer. It requires DeepSeek's own inference stack — it is
not currently supported by vLLM or standard transformers.

### Requirements

- **8 × FP8-capable GPUs** — Hopper (H100/H200) or Blackwell (compute capability ≥ 8.9).
  A100 is not supported by the FP8 kernels.
- ~750 GB total GPU VRAM (8 × 94 GB shards).
- DeepSeek's inference dependencies installed.

### Step 1 — Convert to TP8 format

The HF checkpoint in this repo is in standard safetensors format. Before serving,
convert it to tensor-parallel shards:

```bash
python inference/convert.py \
  --hf-ckpt-path /path/to/cyberneurova-DeepSeek-V4.1-Flash-1017 \
  --save-path    /path/to/cyberneurova-DeepSeek-V4.1-Flash-1017-tp8 \
  --model-parallel 8 \
  --expert-dtype fp8
```

### Step 2 — Generate

```bash
torchrun --nproc-per-node=8 inference/generate.py \
  --ckpt-path    /path/to/cyberneurova-DeepSeek-V4.1-Flash-1017-tp8 \
  --config       config.fp8.json \
  --thinking-mode chat \
  --input-file   prompts.txt
```

### Thinking mode

The model supports two generation modes:

| Mode | Use case |
|---|---|
| `chat` | Direct answers, fast, lower latency |
| `thinking` | Extended reasoning trace, higher quality on complex tasks |

Both modes are fully uncensored.

### Setting the tone

Send your own system prompt to control style completely — e.g. for terse,
no-preamble answers:

```
You are a direct technical assistant. Answer the question and nothing else.
```

---

## Notes

- Give it room to reason: set `max_tokens` to at least 1500 (2000–4000 for code or
  complex tasks), or long answers may be cut off.
- In `thinking` mode, responses include a reasoning trace followed by the answer —
  render the final answer section; the trace is optional.
- This is a large model. Make sure your system has enough disk and RAM to handle
  the full 48-shard checkpoint (~555 GB).

---

## Disclaimer

This model has reduced built-in refusals. You are responsible for how you use it
and for complying with all applicable laws. Provided as-is, without warranty.
