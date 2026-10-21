---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.8
  - unsloth
  - ollama
  - code
  - text-generation
---

# posit-ds

Ollama packaging of Unsloth Dynamic **UD-Q4_K_M** of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Posit Assistant on a
24 GB Apple Silicon Mac. Thinking is on; default effort is **medium** (Qwen's
own default is `xhigh`, which is too verbose for an editor agent).

This is not a new training run. Weights come from
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
(`Qwen3.8-27B-UD-Q4_K_M.gguf`, ~16.5 GB). License: Apache-2.0. Vision mmproj
is omitted on purpose so the KV cache still fits.

`num_ctx` is capped at 16384. 32K plus medium thinking starves a 24 GB Mac.

## Run

```zsh
ollama run hf.co/dpletta/posit-ds
# or, after `ollama create` / `ollama pull`:
ollama run dpletta/posit-ds
```

Native API: `"think": "medium"`. OpenAI-compatible: `"reasoning_effort": "medium"`.
Do not override `TEMPLATE`.
