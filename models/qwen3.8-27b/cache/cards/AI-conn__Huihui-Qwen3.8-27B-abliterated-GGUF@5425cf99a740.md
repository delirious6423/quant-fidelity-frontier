---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - uncensored
  - abliterated
  - ollama
---

# Huihui-Qwen3.8-27B-abliterated-GGUF — AI-conn fork

> **This is a fork, not our model.** All credit for the abliterated build goes
> to **[huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF)**,
> and for the base model to **Qwen (Qwen/Qwen3.8-27B)**.
> This fork exists so the fleet's copy cannot be altered or pulled upstream,
> and to carry the fit notes below. License: Apache-2.0 (per the source repo).

The build was made by **abliteration** (refusal-direction removal, using
Sumandora's remove-refusals-with-transformers) — the method and tool are
documented on the source card. Note from the same card: several sub-series
(Swift / Saluki / Ternary variants) ablate only layers 22–52; the headline
UD-DW and plain series are the main line.

**Recommended file for a 20 GB card:** `Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf` (15.41 GiB).

**Ollama pull:**

```
ollama pull hf.co/AI-conn/Huihui-Qwen3.8-27B-abliterated-GGUF:Q4_K_M
```

## Prime fit (measured budget, RX 7900 XT — 19.98 GiB VRAM, 32 GB RAM)

Fit math (weights + fp16 KV at 0.125 MiB/token + ~0.7 GiB runtime reserve):

- UD-DW-Q4_K_M 15.41 GiB + 8k ctx (1.0 GiB) ≈ **17.1 GiB — fully resident**
- Same weights + 16k ctx (2.0 GiB) ≈ **18.1 GiB — fully resident**

This is the only top-tier uncensored build in our audit with headroom for
16k context on a 20 GB card, which is why it is the fleet's primary brain.
For comparison: any 32B-class Q4 build (stock or abliterated) leaves <1 GiB
for KV + compute buffers on this card and partially CPU-offloads by
arithmetic, whatever the card claims.
