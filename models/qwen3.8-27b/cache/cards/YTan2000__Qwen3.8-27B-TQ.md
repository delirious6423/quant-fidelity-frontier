---
license: apache-2.0
language:
- en
library_name: gguf
pipeline_tag: text-generation
model_name: Qwen3.8-27B-TQ
tags:
- gguf
- qwen
- qwen3.8
- tqllm
- speculative-decoding
- long-context
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-TQ

Engine: [tqLLM](https://tqllm.com) 1.0 (`docker pull ytan2000/tqllm:1.0.0` or `ghcr.io/turbo-tan/tqllm:1.0.0`).

Qwen3.8-27B for the tqLLM engine in four sizes (11.85 to 14.72 GB), with a DFlash2 draft model for exact
speculative decoding. TQ_XS and TQ_S target 16 GB GPUs; TQ_L targets 24 GB GPUs and GB10.

## Credits

| What is in these files | Credit |
|---|---|
| Qwen3.8-27B weights, MTP head, chat template | Qwen team, Alibaba ([Qwen](https://huggingface.co/Qwen)) |
| DFlash2 draft model weights and method | Jian Chen, Yesheng Liang, Zhijian Liu, [z-lab](https://github.com/z-lab/dflash) ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) |
| BF16 source GGUF | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth) |
| Fixed chat template (embedded) | froggeric, [Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) |
| Rotation-based quantisation, the lineage of the TQ3 / TQ4 formats | Google Research, [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) |

Credits for the engine (KV cache, speculative decoding, kernels): [tqllm.com/credits](https://tqllm.com/credits).

## Files

| File | Size | What it is |
|---|---|---|
| `Qwen3.8-27B-TQ_XS.gguf` | 11.85 GB | smallest; 16 GB GPUs with the most room for context |
| `Qwen3.8-27B-TQ_S.gguf` | 12.55 GB | 16 GB GPUs; closer to BF16 than TQ_XS |
| `Qwen3.8-27B-TQ_M.gguf` | 13.14 GB | experimental TQ3 / TQ4 mix; not in the quality-gated set (KLD only, no task scores) |
| `Qwen3.8-27B-TQ_L.gguf` | 14.72 GB | 24 GB GPUs and GB10; closest to BF16 (TQ4 tensors) |
| `drafters/Qwen3.8-27B-DFlash2-TQ3.gguf` | 0.79 GB | DFlash2 draft model, TQ3 weights |
| `mtp-vocab/atx_65536.txt` | 0.4 MB | the draft model's token list (65,536 ids) |
| `drafters/NOTICE` | | licence notice for the draft model |

Pick one model file (TQ_S shown) and download it with the drafter and its token list:

```sh
hf download YTan2000/Qwen3.8-27B-TQ Qwen3.8-27B-TQ_S.gguf drafters/Qwen3.8-27B-DFlash2-TQ3.gguf mtp-vocab/atx_65536.txt --local-dir models
```

For another size, replace `Qwen3.8-27B-TQ_S.gguf` with `Qwen3.8-27B-TQ_XS.gguf`, `Qwen3.8-27B-TQ_M.gguf` or
`Qwen3.8-27B-TQ_L.gguf` (also in `MODEL_FILE` below).

## Run

These files need tqLLM: their TQ3 / TQ4 tensors use tqLLM's own GGUF types (300, 301), so stock llama.cpp, Ollama and
LM Studio cannot load them.

```sh
docker run --rm --gpus all -p 127.0.0.1:8086:8086 -v "$PWD/models:/models:ro" \
  -e MODEL_FILE=/models/Qwen3.8-27B-TQ_S.gguf \
  -e DRAFT_MODEL_FILE=/models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
  -e DRAFT_VOCAB_FILE=/models/mtp-vocab/atx_65536.txt \
  -e KV_FORMAT=tq3 -e CONTEXT=32768 -e REASONING=on -e MAX_TOKENS=4096 \
  ytan2000/tqllm:1.0.0
```

The image runs on x86-64 (RTX 30 / 40 / 50) and arm64 (GB10 / DGX Spark). The server is OpenAI-compatible
(`/v1/chat/completions`). All options: `docker run --rm --gpus all --entrypoint tqserve ytan2000/tqllm:1.0.0 --help`.

- **Drafter:** the draft model proposes several tokens and the 27B checks them in one pass; every accepted token is
  exact, so the output is the model's own text, only faster.
- **Long context:** `KV_FORMAT=tq3` stores the KV cache at about 3 bits per value. On 16 GB cards TqCache keeps the
  oldest KV in system RAM when the requested context does not fit on the GPU (exact, at a decode cost from PCIe).

## Quality

| | TQ_XS | TQ_S | TQ_M | TQ_L |
|---|---:|---:|---:|---:|
| Size | 11.85 GB | 12.55 GB | 13.14 GB | 14.72 GB |
| KLD vs BF16 (mean per-token KL divergence, lower is closer) | 0.079 | 0.055 | 0.045 | 0.022 |
| Same top token as BF16 | 88.6 % | 90.1 % | 91.2 % | 93.7 % |
| GSM8K (1319, greedy, thinking on) | 95.4 % \* | 94.6 % | - | 94.9 % |
| MMLU-Pro (291-question subset) | 78.7 % \* | 83.2 % | - | 83.5 % |
| GPQA Diamond (198) | 84.85 % | 82.83 % | - | 87.88 % |

KLD on GB10. GSM8K / MMLU-Pro on GB10, except \* on RTX 3090; GPQA on RTX 3090. tqLLM release engine, DFlash2-TQ3
drafter, tq3 KV. MMLU-Pro on 291 questions has a standard error of about 2.4 points; GSM8K differences between the files
are within noise. TQ_M has KLD only.

BenchLoop by task (RTX 3090, same engine):

| Task | TQ_XS | TQ_S | TQ_L |
|---|---:|---:|---:|
| speed | 92.6 | 93.2 | 91.9 |
| toolcall | 86.7 | 86.7 | 86.7 |
| coding | 79.2 | 85.4 | 87.5 |
| dataextract | 92.9 | 89.0 | 88.4 |
| instructfollow | 76.7 | 91.1 | 84.5 |
| reasonmath | 80.0 | 80.0 | 80.0 |

## Speed (LocalMaxxing canonical prompts, DFlash2-TQ3 drafter, 32K context, thinking on, median tok/s)

| Hardware | TQ_XS reasoning / code | TQ_S reasoning / code | TQ_L reasoning / code |
|---|---:|---:|---:|
| RTX 3090 24 GB | 271.3 / 149.8 | 251.7 / 173.4 | - |
| RTX 5060 Ti 16 GB (Windows 11 / WSL 2) | 175.9 / 98.8 | 166.5 / 109.0 | - |
| GB10 (DGX Spark) | 99.7 / 67.6 | 92.8 / 60.4 | 72.4 / 61.0 |

All rows use the drafter (speculative decoding, exact output). Median of 5 runs (RTX 3090: 3 runs). "-" = not measured
yet. Decode speed with a drafter depends on the text: code and copy-heavy answers accept more drafts than open prose.

## Format

TQ3 (3.125 bits per weight) and TQ4 (4.125) rotate each 128-weight block and store one FP16 scale per block plus 8- or
16-level codes; the rotation spreads outliers so one scale per 128 weights is enough. For comparison, TQ3_4S (the format of
[YTan2000/Qwen3.8-27B-TQ3_4S](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ3_4S)) uses 32-weight blocks with four 8-bit scales, 4.0 bits per weight.

## Chat template

The file embeds the fixed Qwen chat template; no `--chat-template-file` is needed.
