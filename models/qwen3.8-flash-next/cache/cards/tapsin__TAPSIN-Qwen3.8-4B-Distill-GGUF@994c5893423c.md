---
license: apache-2.0
language:
- en
- tr
base_model:
- Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-4B — GGUF

**Developed by [TAPSIN](https://tapsin.net)**

GGUF quantizations of **[tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF](https://huggingface.co/tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF)** — a full-parameter distillation of **Qwen3.8 2.4T A95B** into the Qwen3.5-4B architecture — for [llama.cpp](https://github.com/ggml-org/llama.cpp), Ollama, LM Studio, Jan, KoboldCpp, and other stock GGUF runtimes.

This card is about choosing a file and running it. The capability writeup, full benchmark results, and best practices live on the **[main model card](https://huggingface.co/tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF)**.

Headline results for the source model (CoT protocols, `lm-evaluation-harness`, identical settings base vs. student):

| Task | Qwen3.5-4B (base) | **Qwen3.8-4B** | Δ |
|---|---:|---:|---:|
| mmlu (CoT, 57 subjects) | 0.354 | **0.553** | **+0.199** |
| gsm8k_cot | 0.850 | 0.785 | −0.065 |

> [!Note]
> Qwen3.5-class models are hybrids: three Gated DeltaNet layers for every full-attention layer. A **recent llama.cpp build with Qwen3.5 / Gated TAPSIN support** is required — older builds will fail to load the architecture.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `Qwen3.8-4B-Q4_K_M.gguf` | Q4_K_M | 2.783 GB | **Recommended.** Best quality/size balance for most users. |
| `Qwen3.8-4B-Q5_K_M.gguf` | Q5_K_M | 3.161 GB | Higher quality at a modest size increase. |
| `Qwen3.8-4B-Q6_K.gguf` | Q6_K | 3.563 GB | Near-lossless. |
| `Qwen3.8-4B-Q8_0.gguf` | Q8_0 | 4.611 GB | Highest-quality quantization. |
| `Qwen3.8-4B-BF16.gguf` | BF16 | 8.666 GB | Full precision reference. |

Sizes are exact decimal GB from the uploaded files (1 GB = 1,000,000,000 bytes).

### What fits on a GPU?

Practical weight-size-based guidance at modest context — the KV cache is the dominant cost at long context and may require offload regardless of weight quant:

| Quant | Guidance |
|---|---|
| Q4_K_M / Q5_K_M | Comfortable on 4–6 GB cards; strong CPU-only option as well. |
| Q6_K / Q8_0 | 6–8 GB recommended. |
| BF16 | 12 GB+. |

## Usage

### llama.cpp

```bash
llama-cli -m Qwen3.8-4B-Q4_K_M.gguf \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  -n 16384 -cnv
```

Use the built-in chat template (`-cnv`). The model is a reasoning model: every answer opens with a `<think>` block, so allow a generous `-n` and strip the `<think>...</think>` span for end users.

### Ollama / LM Studio / Jan / KoboldCpp

Download the GGUF of your choice and load it directly; the chat template is embedded in the file. Recommended sampling: `temperature=0.6, top_p=0.95, top_k=20`.

## Provenance & licensing

Quantizations of **[tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF](https://huggingface.co/buckets/tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF)**, a distillation of Qwen3.8 2.4T A95B into [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) trained on ~45,000 curated teacher traces from our internal Qwen3.8 distillation datasets. Weights are **Apache-2.0**, inherited from the Qwen base, shared as-is.

## Stay in the loop

Sign up for the Empero newsletter at **[tapsin.net](https://tapsin.net)** for releases, evals, and research notes.

## Support / Donate

If this model helped you, consider supporting the project:

- **TRC20 - USDT**: `TYCK6ZyMS6UDt787foPH2QwFuvkdqMw1Jv`
- **BSC USDT**: `0x15aac92a1945ddbe5c79304bfa388d6be99b26a3`

## Acknowledgements

- Developed and released by [TAPSIN](https://tapsin.net)
- Base model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Alibaba Qwen team)
- GGUF quantization: [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml-org)