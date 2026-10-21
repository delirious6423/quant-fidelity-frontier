---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- qwen3
base_model: Qwen/Qwen3.8-27B
---

# leonsarmiento/Qwen3.8-27B-3bit-mlx

This model [leonsarmiento/Qwen3.8-27B-3bit-mlx](https://huggingface.co/leonsarmiento/Qwen3.8-27B-3bit-mlx) was
converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
using mlx-vlm — **full multimodal (VLM), vision preserved at 8-bit**.

**Two variants of this build — pick by your RAM budget:**

| | **This model (clean 3-bit)** | [Qwen3.8-27B-3bit-mtp-mlx](https://huggingface.co/leonsarmiento/Qwen3.8-27B-3bit-mtp-mlx) |
|---|---|---|
| Footprint | **12.71 GB** | 13.04 GB (+0.33 GB) |
| Peak memory (pp1024/tg128) | **13.16 GB** | 15.69 GB with MTP on / 13.16 GB with MTP off |
| Speculative decoding | — | ✅ bundled MTP, toggleable |
| Best for | The RAM-poor — leanest footprint, best tok/s per GB | ~3 GB headroom → extra decode speed (+30.7% on oMLX) |

> The MTP variant's Lightning speedup is an **oMLX-only** feature (v0.6.1+); LM Studio
> does not execute bundled MTP layers. **LM Studio users: this clean build is the one you want.**

> **⚠️ Reasoning effort is baked to `low` in this build.** The chat template ships with
> `reasoning_effort|default('low')` patched into both `chat_template.jinja` and
> `tokenizer_config.json` (upstream default is `'xhigh'`). Out of the box, thinking is
> brief and focused — faster TTFT, shorter reasoning chains, less overthinking. For hard
> problems, pass `reasoning_effort='medium'` or `'xhigh'` explicitly (LM Studio: template
> kwargs; oMLX: `chat_template_kwargs`).

**Qwen3.8-27B** is architecturally identical to Qwen3.6-27B (config.json differs only in
`transformers_version`): a dense hybrid-attention VLM with 64 layers (48 linear attention +
16 full attention), 262K context, and an MTP prediction layer. This quantization uses the same
recipe that made [Qwen3.6-27B-3bit-mlx](https://huggingface.co/leonsarmiento/Qwen3.6-27B-3bit-mlx)
the go-to 3-bit build for this family.

## Quantization Details

The model uses mixed quantization:
- **Embedding layers**: 4-bit with group_size=64
- **Prediction head (lm_head)**: 4-bit with group_size=64
- **Vision tower**: 8-bit with group_size=64
- **All other layers**: 3-bit with group_size=64

**Stats**: 3.716 bits per weight · 12.71 GB · 3 shards · mlx-vlm 0.6.13 · vision tower preserved (8-bit, 84 modules) · embeddings + untied lm_head at 4-bit · 496 language modules at 3-bit

## Benchmarks vs vanilla 4-bit uniform (mlx-community/Qwen3.8-27B-4bit)

Both run on [oMLX](https://github.com/jundot/omlx) (Engine: Auto, Context: Code/Python).

**Summary**: this 3-bit build decodes **+7.8% faster** (19.3 vs 17.9 tok/s @ batch 1), uses **3.4 GB less peak memory** (15.6 vs 19.1 GB), and matches the 4-bit uniform on quality (MMLU −2pp, MMLU_PRO +2pp).

### Single Request

**Qwen3.8-27B-3bit-mlx (this model)**

| Test | TTFT (ms) | TPOT (ms) | pp TPS | tg TPS | E2E (s) | Throughput | Peak Mem |
|---|---|---|---|---|---|---|---|
| pp1024/tg128 | 9966.9 | 52.26 | 102.7 tok/s | 19.3 tok/s | 16.614 | 69.3 tok/s | **15.61 GB** |
| pp4096/tg128 | 39974.9 | 63.99 | 102.5 tok/s | 15.8 tok/s | 48.118 | 87.8 tok/s | 16.69 GB |

**Qwen3.8-27B-4bit (mlx-community, vanilla uniform)**

| Test | TTFT (ms) | TPOT (ms) | pp TPS | tg TPS | E2E (s) | Throughput | Peak Mem |
|---|---|---|---|---|---|---|---|
| pp1024/tg128 | 10289.6 | 56.44 | 99.5 tok/s | 17.9 tok/s | 17.469 | 65.9 tok/s | 19.06 GB |
| pp4096/tg128 | 38404.9 | 54.15 | 106.7 tok/s | 18.6 tok/s | 45.295 | 93.3 tok/s | 20.28 GB |

### Continuous Batching (pp1024/tg128)

| Batch | 3bit tg TPS | 4bit tg TPS | 3bit Speedup | 4bit Speedup | 3bit Peak-ish E2E | 4bit E2E |
|---|---|---|---|---|---|---|
| 1x | **19.3 tok/s** | 17.9 tok/s | 1.00x | 1.00x | 16.614 | 17.469 |
| 2x | 37.4 tok/s | 36.8 tok/s | 1.94x | 2.06x | 34.730 | 34.383 |
| 4x | 64.7 tok/s | 66.2 tok/s | 3.35x | 3.70x | 70.509 | 69.697 |

### General Knowledge — Instruct Mode

| Benchmark | Sampled | Qwen3.8-27B-3bit-mlx | Qwen3.8-27B-4bit |
|---|---|---|---|
| MMLU | 50/14042 | 76.0% | **78.0%** |
| MMLU_PRO | 50/12032 | **58.0%** | 56.0% |

### General Knowledge — Reasoning Mode (effort low)

Coming soon.

## Use with mlx

```bash
pip install mlx-vlm
```

## Recommended Inference Parameters - Add to Jinja template on LM studio or Chat Template Kwargs on oMLX

### Template Flags

This model's chat template accepts three flags, set in the jinja template (LM Studio) or as Chat Template Kwargs (oMLX):

- **`enable_thinking`** — `true` (thinking mode, default) / `false` (instruct mode)
- **`preserve_thinking`** — `true` keeps prior assistant reasoning in multi-turn history
- **`reasoning_effort`** — `'low'` / `'medium'` / `'xhigh'` (this build ships with the template default patched to `'low'` for snappy responses — pass `'xhigh'` explicitly for hard problems)

### Sampling Parameters

**Thinking mode for general tasks** (`enable_thinking=true`):
`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`

**Thinking mode for precise coding tasks (e.g. WebDev)** (`enable_thinking=true`):
`temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`

**Instruct (or non-thinking) mode** (`enable_thinking=false`):
`temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`

### Example Usage

```python
from mlx_vlm import load, generate

model, processor = load("leonsarmiento/Qwen3.8-27B-3bit-mlx")

messages = [{"role": "user", "content": "hello"}]

prompt = processor.tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False,
    enable_thinking=True, reasoning_effort="medium",
)

response = generate(model, processor, prompt=prompt, max_tokens=512)
```

### Example with Custom Parameters

```python
from mlx_vlm import load, generate

model, processor = load("leonsarmiento/Qwen3.8-27B-3bit-mlx")

messages = [{"role": "user", "content": "hello"}]
prompt = processor.tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False,
    enable_thinking=False,
)

response = generate(
    model,
    processor,
    prompt=prompt,
    temperature=0.7,
    top_p=0.80,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
    max_tokens=512,
)
```
