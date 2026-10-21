---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - deepseek
  - deepseek-v4
  - mixture-of-experts
  - imatrix
  - single-gpu
  - rtx-pro-6000
  - blackwell
  - 96gb
  - speculative-decoding
  - tool-use
  - reasoning
quantized_by: prometheusAIR
---

# DeepSeek-V4-Flash-0731 GGUF — one RTX PRO 6000 96GB, no offload

Three imatrix-guided quants, plus the **DSpark draft head** for speculative
decoding. Every variant keeps all tensors on a single 96 GB card — no CPU expert
offload, no second GPU, no layer spill.

## Which one do I want?

| | **IQ2_XS** | **IQ2_XXS** | **IQ2_XXS-slim** |
|---|---|---|---|
| Size | 89.93 GiB | 83.15 GiB | **78.84 GiB** |
| Max context tested | 384K | **1M** | **1M** |
| Pairs with DSpark draft | no room | no room | **yes** |
| Decode, short prompt | ~75 tok/s | ~82 tok/s | **~139 tok/s with DSpark** |
| Decode at ~1M | — | ~25 tok/s | **~57 tok/s with DSpark** |
| Mean KL vs master | **0.492** | 0.571 | 0.626 |
| Top-1 agreement | **76.89%** | 74.99% | 73.96% |

- **IQ2_XS** — best weights, if you do not need past ~384K.
- **IQ2_XXS** — full 1M at maximum quality.
- **IQ2_XXS-slim + DSpark draft** — full 1M at ~1.7x decode. Recommended daily driver.

The three form one monotonic ladder: each step down costs roughly 1 point of
top-1 agreement with the unquantized master. Slim gives up 1.03 points against
IQ2_XXS to free the 4.3 GiB the draft head needs.

## Recipes

Imatrix-guided (606,208 calibration tokens at 8K context, 100% expert-slice
coverage). None use `IQ1_S`/`IQ2_S`/`IQ3_S`, whose CUDA kernels are broken on
Blackwell (sm_120).

| tensor group | IQ2_XS | IQ2_XXS | IQ2_XXS-slim |
|---|---|---|---|
| Expert gate/up | `IQ2_XS` | `IQ2_XXS` | `IQ2_XXS` |
| Expert down | `IQ3_XXS` | `IQ3_XXS` | `IQ3_XXS` on blocks 0–9, 33–42; `IQ2_XS` on 10–32 |
| Dense/attention | `Q8_0` | `Q6_K` | `Q6_K` |
| Embedding/output | `Q8_0` | `Q8_0` | `Q8_0` |

Slim keeps full precision on the first and last ten blocks, where quantization
damage concentrates, and spends the savings only in the middle of the stack.

`general.file_type` reports the base type passed to `llama-quantize`, not the
mixture actually present — trust the table, not that field.

## Settings for one RTX PRO 6000 96GB

Needs a llama.cpp build with DeepSeek-V4 support. The draft additionally needs
**b10236 or newer** (upstream #25784 and #26452).

Recommended — **slim + DSpark at full 1M**:

```bash
CUDA_VISIBLE_DEVICES=0 llama-server \
  --model DeepSeek-V4-Flash-0731-IQ2_XXS-slim-00001-of-00004.gguf \
  --spec-draft-model DeepSeek-V4-Flash-0731-DSpark-Q2_K.gguf \
  --spec-type draft-dspark --spec-draft-ngl 99 --spec-draft-n-max 3 \
  --device CUDA0 --split-mode none -ngl 999 --fit off \
  --ctx-size 1048576 -fa on -b 2048 -ub 512 -np 1 \
  --jinja --temp 1.0 --top-p 1.0 --min-p 0.0
```

Point llama.cpp at the first shard; it finds the rest automatically.

| variant | context | `-ub` | free VRAM after load |
|---|---:|---:|---:|
| slim + draft | 1M | 512 | 1,399 MiB |
| IQ2_XXS | 1M | 1024 | 1,752 MiB |
| IQ2_XS | 384K | 1024 | 998 MiB |
| IQ2_XS | 256K | 2048 | 1,234 MiB |
| IQ2_XS | 64K | 2048 | 3,302 MiB |

`-ub 2048` does **not** fit at 384K on IQ2_XS, and `-ub 1024` does not fit at 1M
with the draft attached. If the card also drives a display, take one step down in
ubatch for roughly 740 MiB more headroom.

### Settings that matter

- **`--spec-draft-n-max 3`, not 5.** The draft advertises `block_size 5`, but
  drafting 3 is 27% faster: n_max 2/3/4/5 → 130/139/113/109 tok/s.
- **The draft must share a GPU with the target.** It borrows the target's
  `tok_embd` and output projection, so putting it on a second device aborts at
  graph allocation.
- **Draft precision does not affect speed.** Q2_K, Q3_K and the native MXFP4
  draft all decode within 1.3% of each other; the smallest is published because
  the freed VRAM is worth more in the target.
- **`--fit off` and `--split-mode none`** make a bad configuration fail loudly
  instead of silently spilling to CPU and costing most of your speed.
- **f16 KV.** Quantized KV is unnecessary at 1M with slim, and costs ~4% decode.
- **Reasoning is always on.** Send `max_tokens` comfortably above any
  `--reasoning-budget`, or you get `finish_reason: length` and empty content
  whether or not the model solved the task.

## Validation

IQ2_XXS-slim at 1M with the draft attached: **27/27 depth probes exact** — verbatim
retrieval plus multi-hop reasoning combining facts planted at 10%, 50% and 90%
depth, at 6K / 32K / 131K / 262K / 524K and **991,046 tokens**. Draft acceptance
does not decay with depth (81% at 990K against 74–87% at 6–32K). IQ2_XXS was
separately validated with exact retrieval from a 989,987-token prompt.

KL divergence is measured against the release-faithful MXFP4/BF16 master over
wikitext-2 (`-c 512 --chunks 100`). Perplexity is deliberately not reported: it
can rank a lower-bit quant above its own higher-bit reference, so it does not
measure quantization damage.

A near-1M prefill takes roughly 55 minutes on this hardware — these are
large-but-not-maximal-prompt configurations, not million-token batch processors.
Once a long context is loaded, follow-up questions reuse the cached prefix and
cost seconds.

## Known issue: reasoning length on recent llama.cpp

On builds including upstream `dee2a846b` (*ggml : adjust logic for offloading ops
to weight's backend*, #25832), this model's reasoning runs roughly 5x longer and
degenerates into repetition — 13,449 reasoning characters versus 2,613 on the
immediately preceding commit, identical prompt at temperature 0. It is a
graph-scheduling change affecting DeepSeek-V4 numerics, bisected to that single
commit, and it reproduces on every quant here. It is unrelated to quantization.
The DSpark draft requires a build newer than this, so the two cannot currently be
separated.

## Files

```bash
sha256sum -c SHA256SUMS
```

| set | files |
|---|---|
| IQ2_XS | `DeepSeek-V4-Flash-0731-IQ2_XS-0000{1..4}-of-00004.gguf` |
| IQ2_XXS | `DeepSeek-V4-Flash-0731-IQ2_XXS-0000{1..4}-of-00004.gguf` |
| IQ2_XXS-slim | `DeepSeek-V4-Flash-0731-IQ2_XXS-slim-0000{1..4}-of-00004.gguf` |
| DSpark draft | `DeepSeek-V4-Flash-0731-DSpark-Q2_K.gguf` (6.43 GiB) |

## Attribution

Base model and architecture by DeepSeek-AI. Quantization, calibration and
single-GPU validation by `prometheusAIR`. This quant inherits the capabilities,
limitations, risks and MIT license of the base model; you are responsible for
evaluating outputs and applying safeguards appropriate to your application.

Please read the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
for architecture details, recommended sampling and the base model's own limitations.

```bibtex
@misc{deepseekai2026deepseekv4,
  title  = {DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author = {DeepSeek-AI},
  year   = {2026}
}
```
