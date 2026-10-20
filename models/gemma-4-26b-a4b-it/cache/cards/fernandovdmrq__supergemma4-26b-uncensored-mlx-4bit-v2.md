---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags:
  - gemma4
  - mlx
  - uncensored
  - apple-silicon
  - 4bit
  - quantized
  - reasoning
  - tool-use
  - coding
  - browser-automation
  - korean
  - fast
language:
  - en
  - ko
pipeline_tag: text-generation
library_name: mlx
---

# SuperGemma4-26B-Uncensored-Fast v2

A faster, sharper, uncensored Gemma 4 26B for Apple Silicon.

This is the text-only flagship for people who want the core trade-off to be obvious at a glance:

- smarter than stock `Gemma 4 26B IT` on real local agent tasks
- faster than the stock local 4-bit baseline on the same machine
- uncensored, without falling apart on code, tool-use, or Korean prompts

## Why this model

If you want the fast line instead of the multimodal line, this is the one to run.

- `Fast` is part of the release identity, not just a minor variant
- Uncensored behavior is preserved while practical capability goes up
- Strong at code, browser tasks, tool-use, planning, and Korean
- Tuned for local agent workloads on Apple Silicon MLX

## Headline numbers

| Metric | Gemma 4 26B IT original 4bit | SuperGemma Fast |
| --- | ---: | ---: |
| Quick bench overall | `91.4` | `95.8` |
| Avg generation speed | `42.5 tok/s` | `46.2 tok/s` |
| Delta overall | baseline | `+4.4` |
| Delta speed | baseline | `+8.7%` |

## Category gains vs original

| Category | Original | SuperGemma Fast | Delta |
| --- | ---: | ---: | ---: |
| Code | `92.3` | `98.6` | `+6.3` |
| Browser | `87.5` | `89.6` | `+2.1` |
| Logic | `86.9` | `95.2` | `+8.3` |
| System Design | `97.8` | `98.9` | `+1.1` |
| Korean | `90.7` | `95.0` | `+4.3` |

## What makes it attractive

- Beats the stock local 4-bit baseline in both quality and speed
- Produces stronger code, stronger reasoning, and more useful tool-oriented answers
- Handles Korean and agent-style prompts better than the original local run
- Keeps the uncensored feel without turning unstable or collapsing into broken outputs
- Built to feel immediately stronger in real usage, not just in a niche benchmark

## Base and format

- Base model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- Format: MLX 4-bit
- Size: about `13GB`
- Best use case: fast text-only local agent model with stronger practical capability than stock Gemma 4

## Why it is better than stock

- Higher quick-bench overall score: `95.8` vs `91.4`
- Faster average generation speed: `46.2 tok/s` vs `42.5 tok/s`
- Bigger gains where local agents actually benefit:
  - Code: `+6.3`
  - Logic: `+8.3`
  - Korean: `+4.3`
  - Browser workflows: `+2.1`
- Uncensored behavior remains a core property of the release instead of being layered on after the fact

## Recommended launch

```bash
mlx_lm.server \
  --model Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2 \
  --port 8080
```

For OpenAI-compatible serving, let `mlx_lm.server` auto-detect the bundled template.

Do not pass `--chat-template /path/to/chat_template.jinja` as a literal path string on launch paths that expect the template body. That can corrupt responses.

## Quick test

```bash
mlx_lm.generate \
  --model Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2 \
  --prompt "Write a Python function that returns prime numbers up to n." \
  --max-tokens 512
```

## Included files

- `benchmark_quick_bench_20260412.json`
- `benchmark_quick_bench_20260412_responses.jsonl`
- `SERVING_NOTES.md`

## Notes

- This is the fast text-only line.
- The earlier "reasoning is broken" report reproduced as a serving-template launch issue, not as weight corruption.
- Re-fused and re-benchmarked locally before upload.
