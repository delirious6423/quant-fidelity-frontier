---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - deepseek
  - moe
  - reap
  - expert-pruning
  - multimodal
---

# DeepSeek-V4.1-Flash-REAP-272E

`deepseek-ai/DeepSeek-V4.1-Flash` with its routed-expert pool pruned **384 → 272 per
layer** (−29.2%) by REAP — router-weighted expert activation pruning — calibrated on
**text *and* images**.

**475.2 GiB → 396.8 GiB.** Nothing is re-quantized: experts stay MXFP4 (E2M1 + E8M0,
block 32), dense layers stay FP8 block 32×32, the Engram tables are untouched. The only
change is that 112 of 384 experts per layer are gone, along with their rows of
`gate.weight`, `gate.bias` and `gate.bias_vl`.

Recipe, calibration scripts and every measurement behind this card:
**[github.com/Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)**

## Why prune instead of quantize

V4.1-Flash **already ships in 4-bit** — `"expert_dtype": "fp4"`, and routed experts are
56.6% of the checkpoint. Converting them to NVFP4 makes the file *bigger* (4.5 bits/weight
against MXFP4's 4.25). With precision already spent, the remaining lever on this
architecture is the expert *count*.

| | size | experts/layer |
|---|---:|---:|
| `deepseek-ai/DeepSeek-V4.1-Flash` | 475.2 GiB | 384 |
| **this repo** | **396.8 GiB** (207.7 GiB here + 189.1 GiB Engram from the base repo) | **272** |
| [`-REAP-256E`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) | 385.5 GiB | 256 |

Backbone parameters fall from ~552B to ~393B. **Active parameters are unchanged** (6
routed + 1 shared per token) — pruning buys memory, not decode speed.

## Measurements

Held-out text perplexity, 65,504 tokens, verified zero overlap with the calibration
corpus:

| experts kept | perplexity | vs unpruned |
|---:|---:|---:|
| 384 (unpruned) | 3.6552 | — |
| **272 (this repo)** | **3.8060** | **+4.13%** |
| 256 | 3.8557 | +5.49% |
| 192 | 4.1910 | +14.66% |

Caption perplexity conditioned on held-out images, loss over caption tokens only
(2,048 caption tokens), at 272 kept, by ranking:

| ranked by | caption ppl | vs unpruned |
|---|---:|---:|
| *(unpruned)* | 24.307 | — |
| **text + image saliency (this repo)** | **24.769** | **+1.9%** |
| random | 26.066 | +7.2% |
| text-only saliency | 28.173 | +15.9% |

No downstream task benchmarks were run. Perplexity is not a benchmark.

## The reason this repo was calibrated on images

The model carries **two** router biases — `gate.bias` and `gate.bias_vl` — and their
correlation is **−0.05** (min −0.45 across layers). Expert specialisation here is
modality-dependent.

Rank experts by text saliency alone and you discard **267 experts** that a combined
ranking keeps, including cases like layer 0 expert 235: **376th of 384 on text, 21st on
images**. The resulting checkpoint is **worse than randomly pruning the same number of
experts** on vision — and its *text* perplexity looks fine, marginally better than the
combined ranking. Only an image-conditioned evaluation catches it.

If you prune this model yourself, calibrate on both modalities.

## Choosing between 272E and 256E

- **272E (this repo)** is the better checkpoint: +4.13% text perplexity, no measurable
  vision regression.
- **[256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E)** exists
  because **vLLM's fused MoE router kernel only accepts expert counts from a fixed
  table** — `{1,2,4,8,16,32,64,128,192,256,320,384,448,512,576}`. 272 is not in it, and
  the engine refuses to start with `Unsupported expert number: 272`. If you are serving
  under vLLM, take 256E.
- 256E is also ranked differently (`--mode balanced`, see the recipe repo). Straight mean
  saliency collapses the vision path below 272 — caption perplexity **30.959** at 256 —
  because the calibration corpus has 5× more text tokens than image tokens and the mean
  score inherits that ratio. Balanced ranking brings it back to **26.479**, but **pays for
  it on text**: 4.1723 against mean-saliency's 3.8557 at the same 256 experts. There is no
  setting at 256 that gets both. **272E needs no such trade** — it is the better checkpoint
  on every axis measured, and the only reason to prefer 256E is that an engine will load it.

## Serving

**This checkpoint does not load under vLLM** (272 is not in the fused router's
expert-count table). The numbers below are from its sibling
[256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) on the same
hardware, and every kernel-level note applies to any DSV4.1 checkpoint on sm_120/121.

Verified on **2× GB10 (sm_121, 120 GB unified each)** under vLLM TP=2 over RoCE, with the
Engram tables read from NVMe rather than resident:

| | |
|---|---|
| decode, c=1 | **13.7 tok/s** (median of 5 warm) |
| KV cache | **1,437,437 tokens** @ 64K ctx — 21.9× concurrency |
| weights | ~94 GiB/rank; Engram 47.2 GiB/rank left on disk |
| `--gpu-memory-utilization` | **0.90** (0.80 leaves no room for KV; above ~0.91 the free-memory check trips) |

Four things this needs that are not obvious:

1. **`--enforce-eager` is mandatory.** With CUDA graphs on, the model loads, serves, and
   returns fluent-looking bytes that are **NaN all the way down** — `logprobs` fails with
   `Out of range float values are not JSON compliant: nan`. Eager costs ~7% here (14.5 →
   13.7 tok/s), which on this hardware is the right trade; GB10 is bandwidth-bound, so
   graphs buy little anyway.
2. **FlashInfer has no sparse-MLA kernel for this model's index widths.** DSV4.1 mixes
   compress ratios, giving topk **1152** and **640**, and the instantiation list is keyed
   on `(num_heads, topk)` — so the entries you need depend on your TP degree
   (TP=2 → 32 heads, TP=4 → 16). Without them the engine dies at profiling with
   *"SM120 sparse-MLA has no decode kernel for this shape"*.
3. **Patching the `.cu` alone does nothing.** The wheel ships a prebuilt AOT
   `sparse_mla_sm120.so` and `is_aot` short-circuits the JIT path, so your edit is never
   compiled. Mask the AOT directory, prebuild into a persistent cache, and mount that
   cache at serve time.

4. **Wire the reasoning parser, or the chain-of-thought lands in `content`.** Thinking is
   on by default and the chat template supplies the opening `<think>`, so with no parser
   you get raw reasoning followed by a bare `</think>` inside `content`, and a `reasoning`
   field sitting `null`. The parser ships and is registered as **`deepseek_v41`** — the
   same name serves both kinds:
   `--reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41`.
   The field is **`reasoning`**, not `reasoning_content`.

The patch and both launch scripts are in the recipe repo.

## Contents — read this before downloading

This repo carries **shards 1–46 only** (208 GiB): the pruned weights, the config, the
tokenizer, and the checkpoint's bundled reference implementation.

**Shards 47–48 are the two Engram tables and are NOT in this repo.** Pruning does not
touch them — they are byte-identical to the base model, 189.1 GiB, and re-uploading them
would cost every user a second copy of data they may already have. Fetch them from the
base model and drop them in:

```bash
hf download LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E --local-dir dsv41-reap272
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir dsv41-reap272 \
   --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors"
```

`model.safetensors.index.json` already references them, so the model loads once both
files are present.

The Engram lookup is a hashed n-gram *gather* — 48 rows per token, ~12.4 KB — not a
matmul. It never needs to be resident, which is why it can be served from disk or host
RAM rather than accelerator memory, and why leaving it out of this repo costs nothing
architecturally.

The FP4 Engram tables from
[`LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4)
(97.6 GiB instead of 189.1, cosine 0.9934) are a size-for-accuracy alternative —
**lossy and untested under any serving engine**, offered as-is.

`config.json` records the provenance:

```json
"pruned_from": {"n_routed_experts": 384, "method": "REAP",
                "calibration": "262,144 text tokens + 256 images, saliency combined"}
```

## Limitations

- Calibration is 262,144 text tokens and 256 images. Larger would be better.
- Perplexity only; no task evaluations.
- The caption evaluation is 2,048 tokens — large enough to resolve the random and
  text-only gaps, not the +1.9% figure above.
- MTP layers are retained and were **not** included in the prune (they carry their own
  128-expert pool, which the 384-expert keep-set does not index).
- Not servable by vLLM at this expert count — see above.

## License

Model: inherits the base model's license. Tooling: MIT.
