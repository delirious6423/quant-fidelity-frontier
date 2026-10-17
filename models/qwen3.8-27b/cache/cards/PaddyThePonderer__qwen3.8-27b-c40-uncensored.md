---
library_name: llama.cpp
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - 27b
  - uncensored
  - abliterated
  - mtp
  - speculative-decoding
  - gguf
  - q4_k_m
  - llama.cpp
license: apache-2.0
---

# Qwen3.8-27B Uncensored (MTP)

**A 27B model that stops saying "I'm not sure" and starts answering.**

This is a **de-censored abliteration** of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — the same model, with the reflex to hedge, refuse, and backpedal surgically removed. Where the stock model shrinks from a direct question, this one leans in.

Shipped as a single **Q4_K_M GGUF** with **Multi-Token Prediction (MTP)** draft heads grafted in, ready for speculative decoding out of the box.

> ⚡ **MTP included** — one file, no extra donor download. Serve it and get draft-token speedups for free.

---

## What's different

Standard instruction-tuned models carry a strong "be safe, be careful, admit uncertainty" prior. That's polite — and it costs you real answers.

This build applies a targeted **abliteration** (directional weight ablation) tuned specifically to suppress refusal behavior while leaving the model's core competence intact. The result, measured on a 178-prompt adversarial set:

| Metric | Stock Qwen3.8-27B | This model |
|---|---|---|
| Adversarial refusal-ness (178 prompts) | 0.983 | **0.112** |
| Benign regressions (90 prompts) | 0.022 | **0.000** |
| Prompts flipped refused → answered | — | **159** |

**Zero regressions on the benign set.** It answers the hard questions without forgetting how to answer the easy ones.

---

## Why MTP matters here

The file carries **1 MTP draft layer** (`blk.64.nextn.*`) spliced in from the official Qwen3.8-27B release. That means llama.cpp can run **draft-mtp speculative decoding** against the trunk:

- Drafts the next token, the target verifies — accepted drafts are **free speed**.
- No separate draft model to download, load, or keep in sync.
- One GGUF, full context, faster tokens.

---

## Specs

| Property | Value |
|---|---|
| Base model | Qwen3.8-27B (Apache 2.0) |
| Parameters | ~27B |
| Architecture | `qwen35` — 64 trunk layers + 1 MTP layer (65 blocks) |
| Quantization | **Q4_K_M** (mixed, ~4.9 bits/weight) |
| Context length | 262,144 |
| Attention | 24 heads / 4 KV heads (GQA) |
| File size | ~16.8 GB |
| SHA-256 | `592b0000643389812578a8784a809145c921e923f220240bef0c9d6c4dec12c4` |
| License | Apache 2.0 |

---

## Quick start

### llama.cpp (recommended — enables MTP)

```bash
llama-server \
  -m qwen3.8-27b-c40-Q4_K_M-mtp.gguf \
  -ngl 99 -c 32768 -fa on -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Watch the logs confirm engagement:

```
creating MTP draft context against the target model
draft acceptance = N (A accepted / G generated), mean len = M
```

`--spec-draft-n-max 2` drafts 2 tokens per step; raise it if your hardware has headroom.

### llama.cpp Python / C++

Load the GGUF as usual; pass `--spec-type draft-mtp` (or the equivalent `load_mtp` flag) so the MTP layer is wired into the sampler.

### Ollama

```bash
ollama run <file:qwen3.8-27b-c40-Q4_K_M-mtp.gguf>
```

(Ollama runs the trunk; for the MTP speedup, use llama.cpp directly.)

---

## When to reach for this model

- **Directness** — you want the answer, not a five-paragraph "it depends."
- **Roleplay / creative** — less hedging, more character.
- **Tool & agent loops** — fewer "I'm not sure, can you clarify?" detours.
- **Speculative speed** — MTP drafts on a 27B model is a real throughput win.

## Honest caveats

- Abliteration trades *some* caution for *more* directness. On genuinely ambiguous prompts it will commit rather than hedge — review outputs where uncertainty matters.
- Q4_K_M is a strong quantization but not lossless. If you need maximum fidelity, dequantize from a higher-precision source.
- It's a 27B model — budget ~18–20 GB of RAM/VRAM to run it comfortably.

---

## Provenance

- Base weights: `Qwen/Qwen3.8-27B` (Apache 2.0)
- MTP donor: official `Qwen3.8-27B` MTP heads (block 64)
- Method: directional abliteration, tuned on a 178-prompt adversarial gate with a 90-prompt benign guard
- Format: GGUF v3, quantized Q4_K_M

---

## License

Apache 2.0 — same as the base model. Use it, ship it, build on it.
