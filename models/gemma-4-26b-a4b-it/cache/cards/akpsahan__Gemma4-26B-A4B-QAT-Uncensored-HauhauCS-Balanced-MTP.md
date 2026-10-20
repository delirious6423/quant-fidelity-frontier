---
license: gemma
tags:
  - uncensored
  - gemma4
  - moe
  - gguf
  - vision
  - multimodal
  - agentic
  - coding
  - creative-writing
  - roleplay
  - rp
  - conversational
language:
  - en
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
---

# Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP

> **[Join the Discord](https://discord.gg/SZ5vacTXYf)** for updates, roadmaps, projects, or just to chat.

Gemma4-26B-A4B (QAT) uncensored by HauhauCS. **0/465 Refusals***

## About

No changes to datasets or capabilities — fully functional, 100% of what the original authors intended, just without the refusals. Built from the official QAT weights, so the 4-bit quant stays close to full-precision quality.

## Balanced

The **Balanced** variant (recommended — 99%+ of users will be happy here) uses optimized full uncensoring tuned especially for agentic coding, reasoning, creative writing and reliability-critical tasks. It reasons before answering and stays dependable and on-instruction. An **Aggressive** variant, for cases where Balanced still deflects too much, after current testing is not required.

## ~35% faster with MTP

Ships with an MTP (multi-token-prediction) draft head for **speculative decoding** — roughly **35% faster generation with identical output** (the model verifies every drafted token, so quality is unchanged — pure speed). This release is tuned to pair well with the included MTP head.

llama.cpp:
```bash
llama-server \
  -m Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
  -md mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp \
  -ngl 99 -fa on
```

**Note:** the MTP speedup was currently tested by me through **llama.cpp** (`llama-server` / `llama-cli`).

## Downloads

| File | Type | Size |
|------|------|------|
| `Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf` | Q4_K_M (text) | 16.8 GB |
| `mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf` | mmproj (vision) | 1.2 GB |
| `mtp-gemma-4-26B-A4B-it.gguf` | MTP speculative drafter | 252 MB |

> **Why only Q4_K_M?** Gemma 4 is quantization-aware-trained for ~4-bit, so Q4_K_M is the sweet spot — higher-precision quants add size with no real quality gain. Carefully quantized for best quality at 4-bit.

## Vision

Load the mmproj alongside the model for image input:
```bash
llama-server -m Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
  --mmproj mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf -ngl 99 -fa on
```

## Recommended sampling

These are dialed in specifically for this HauhauCS build — use them for the intended behaviour and quality:

- `temperature 0.6`
- `top_k 64`
- `top_p 0.9`
- `min_p 0.05`
- `repeat_penalty 1.1`

This release is tuned end-to-end as its own thing; the settings above are part of that and aren't the stock Gemma defaults.

## Specs

- 26B-A4B MoE (128 experts, 8 active per token) · 256K (262144) context
- Vision (image input) via mmproj
- Based on [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) by Google DeepMind

## Compatibility

- Works with llama.cpp, LM Studio, Jan, koboldcpp, and other GGUF runtimes.
- **Multi-GPU + LM Studio:** I've personally noticed Gemma 4 can crash under LM Studio's *tensor-split* mode — use a single GPU (layer-split or priority order) for this model.

## Acknowledgements

- **Google DeepMind** — Gemma 4.
- The included `mtp-gemma-4-26B-A4B-it.gguf` speculative draft head comes from **Unsloth**'s Gemma 4 release — many thanks to the Unsloth team for it.


\* _Tested with both automated and manual refusal benchmarks — none have been found in standard use. A small number of edge-case prompts deflect on the first ask but comply on a re-ask or strategic framing. If you hit one that's actually obstructive to your use case, [join the Discord](https://discord.gg/SZ5vacTXYf) and flag it so I can work on it in a future revision._
