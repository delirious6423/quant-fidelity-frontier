---
license: apache-2.0
tags:
  - uncensored
  - gemma4
  - moe
  - gguf
  - vision
  - multimodal
  - agentic
  - coding
language:
  - en
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
---

> **Mirror.** This repository is an unmodified copy of a subset of [HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced](https://huggingface.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced), kept as a backup. Included: Q8_K_P and the f16 mmproj only. All credit belongs to the original authors; the original model card follows.


# Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced

> **[Join the Discord](https://discord.gg/SZ5vacTXYf)** for updates, roadmaps, projects, or just to chat.

Gemma4-26B-A4B uncensored by HauhauCS. **0/465 Refusals*** **Release Candidate after over 1 month of nonstop work on this one.**

> **HuggingFace's "Hardware Compatibility" widget doesn't recognize K_P quants** — it may show fewer files than actually exist. Click **"View +X variants"** or go to **Files and versions** to see all available downloads.

## About

**GenRM Defeated!**

No changes to datasets or capabilities. Fully functional, 100% of what the original authors intended — just without the refusals.

These are meant to be the best lossless uncensored models out there.

## Balanced — Release Candidate

This legitimately took me over 1 month of non-stop work. Targeting 0 refusals in standard use, and that's what I'm seeing in testing (automated and manual) — a handful of edge-case prompts still deflect on first try but **follow through on a re-ask**. If you hit one Balanced won't get past, the Aggressive variant is coming once I figure out how to maintain lossless/near-lossless quality for it.

- **Balanced**: will reason through edgy requests, occasionally attach a short safety framing, then deliver the full answer. Output is complete, nothing held back, but it can talk itself into it first. **Recommended default — 99%+ of users will be happy here.** Best for **creative writing, RP, emotional intelligence**. Normally I'd also say "agentic coding/tool use" however in my in-depth testing, **Qwen3.6 has been net superior on such tasks**. Do be mindful of the few deflection categories I mentioned already.
- **Aggressive** *(separate release, WIP)*: strips the self-reasoning preamble and gives direct answers to any DEEPLY censored topics.

Balanced also has meaningfully more stable sampling across re-runs, which matters for long context sessions — no sporadic topic drift deep. 

## Downloads

| File | Quant | BPW | Size |
|------|-------|-----|------|
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q8_K_P.gguf | Q8_K_P | 8.64 | 27 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q6_K_P.gguf | Q6_K_P | 7.21 | 23 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q5_K_P.gguf | Q5_K_P | 6.12 | 19 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q5_K_M.gguf | Q5_K_M | 6.06 | 19 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf | Q4_K_P | 5.36 | 17 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf | Q4_K_M | 5.32 | 17 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf | IQ4_XS | 4.41 | 14 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_P.gguf | Q3_K_P | 4.25 | 13 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf | Q3_K_M | 4.21 | 13 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf | IQ3_M | 3.93 | 12 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf | Q2_K_P | 3.39 | 11 GB |
| Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ2_M.gguf | IQ2_M | 3.29 | 10 GB |
| mmproj-Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-f16.gguf | mmproj (f16) | — | 1.2 GB |

BPW is slightly higher than nominal across the board because Gemma4 has a lot of per-layer norm/scale tensors kept at F32 (multiple post-ffw norms per layer). All quants generated with importance matrix (imatrix) for optimal quality preservation on uncensored weights.

## What are K_P quants?

K_P ("Perfect") quants are HauhauCS custom quantizations that use **model-specific** analysis to selectively preserve quality where it matters most. Each model gets its own optimized quantization profile — the top 25% most-important tensors (per imatrix calibration) are promoted to a higher quant type.

A K_P quant effectively bumps quality up by 1-2 quant levels at only ~5-15% larger file size than the base quant. Fully compatible with llama.cpp, LM Studio, and any GGUF-compatible runtime — no special builds needed.

**Note:** K_P quants may show as "?" in LM Studio's quant column. This is a display issue only — the model loads and runs fine.

## Why this model for agentic work

26B total params with only ~4B active per forward pass (top-8 of 128 experts). You get the reasoning footprint of a 26B with the throughput of a ~4B for inference cost — which matters when you're chaining 10+ tool calls per task. Sliding-window attention (1024 tokens) plus periodic full attention keeps long contexts cheap without losing global coherence.

Balanced is calibrated for this. It removes refusals on security/ops/research-adjacent topics that block legitimate coding work, without bending the sampling geometry that keeps long chains coherent.

Recommended quant for most coding work: **Q4_K_P** (17 GB, fits in 24 GB VRAM with room for context) or **Q8_K_P** (27 GB) if you have more VRAM and want maximum quality with minimal offloading.

Do note - main usecase for Gemma4 is Creative Writing, Roleplaying and Emotional Intelligence.

## Specs

- 25.2B total / 3.8B active params (128 routed experts, top-8 + 1 shared expert)
- 30 layers, hybrid attention: 5× sliding-window (1024 tokens) → 1× full global, repeating. Uses Proportional RoPE (p-RoPE).
- Hidden dim 2816, FFN dim 2112, MoE expert FFN 704, vocab 262144
- Head dim 256 (SWA) / 512 (full), 16 attention heads, 8 KV heads (2 for full layers)
- 256K native context
- Natively multimodal (text + vision) — ships with mmproj. Variable visual token budgets: 70 / 140 / 280 / 560 / 1120 per image.
- Based on [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)

## Recommended Settings

From the official Gemma authors:

**Inference parameters:**
- `temperature=1.0, top_p=0.95, top_k=64`


**Important:**
- Use `--jinja` with llama.cpp for proper chat template handling
- Vision support requires the `mmproj` file alongside the main GGUF. **Place images before text** in your prompt for best vision performance.
- Keep at least 32K context for serious agentic work; the model can take much more (256K native) if you need it
- Sliding window is baked into the architecture — no special flag needed

## Turning Thinking On/Off

Gemma4 has thinking mode controlled via `enable_thinking` in the chat template. It's the same pattern as Qwen3.6 — set `false` for faster, shorter replies and `true` (default) when you want chain-of-thought.

### LM Studio

1. Load the model
2. Right-side settings panel → **Model Settings** → **Prompt Template** (or **Chat Template Options**)
3. Set `enable_thinking` to `false` (or `true`) in the template kwargs

### llama.cpp

**llama-server — set as default for all requests:**
```bash
llama-server -m Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf \
  --mmproj mmproj-Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-f16.gguf \
  --jinja -c 32768 -ngl 99 \
  --chat-template-kwargs '{"enable_thinking": false}'
```

**Per-request via the OpenAI-compatible API:**
```json
{
  "model": "gemma4-26b-a4b",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## Usage

Works with llama.cpp, LM Studio, Jan, koboldcpp, and other GGUF-compatible runtimes.

**llama-server:**
```bash
llama-server -m Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf \
  --mmproj mmproj-Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-f16.gguf \
  --jinja -c 32768 -ngl 99
```

**llama-cli:**
```bash
llama-cli -m Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf \
  --mmproj mmproj-Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-f16.gguf \
  --jinja -c 32768 -ngl 99
```

## Other Models

- [HauhauCS on HuggingFace](https://huggingface.co/HauhauCS/models)

---

\* _Tested with both automated and manual refusal benchmarks — none have been found in standard use. A small number of edge-case prompts deflect on the first ask but comply on a re-ask or strategic framing. If you hit one that's actually obstructive to your use case, [join the Discord](https://discord.gg/SZ5vacTXYf) and flag it so I can work on it in a future revision._
