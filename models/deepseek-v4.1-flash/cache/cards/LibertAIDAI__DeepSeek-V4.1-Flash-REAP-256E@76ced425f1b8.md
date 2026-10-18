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
  - vllm
---

# DeepSeek-V4.1-Flash-REAP-256E

`deepseek-ai/DeepSeek-V4.1-Flash` with its routed-expert pool pruned **384 → 256 per
layer** (−33.3%) by REAP — router-weighted expert activation pruning — using a
**modality-balanced** ranking calibrated on text *and* images.

**475.2 GiB → 385.5 GiB.** Nothing is re-quantized: experts stay MXFP4 (E2M1 + E8M0,
block 32), dense layers stay FP8 block 32×32, the Engram tables are untouched. 128 of
384 experts per layer are gone, along with their rows of `gate.weight`, `gate.bias` and
`gate.bias_vl`.

**This is the checkpoint to use under vLLM.** Its sibling
[`-REAP-272E`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E) scores
better but does not load — see below.

Recipe, calibration scripts and every measurement behind this card:
**[github.com/Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)**

## Why 256 and not 272

vLLM's fused MoE router kernel dispatches on a **fixed table of expert counts** —
`{1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576}`. Anything else fails
at startup with `Unsupported expert number: 272`. 256 is the largest legal value below
384, so it is the only prune ratio in this range that an engine will actually run.

That constraint has a cost, and it is not the one text perplexity shows.

## Why the ranking is different from 272E — and what it costs

The default REAP score is the **mean** router-weighted activation norm over the
calibration set. Our corpus is 64.4M text tokens against 13.0M image tokens, so the mean
inherits that 5:1 ratio and the ranking is quietly text-heavy. At 272 kept it survives
anyway. Below 272 the vision path falls off a cliff that text perplexity gives no
warning of:

| experts kept, mean-saliency ranking | text ppl | caption ppl |
|---:|---:|---:|
| 384 (unpruned) | 3.6552 | 24.307 |
| 272 | 3.8060 | 24.769 |
| 264 | — | 30.070 |
| 256 | 3.8557 | **30.959** |

Text walks smoothly from +4.1% to +5.5% across that range. Captions jump **+27%**.

The fix is to normalise the text and image saliency vectors to equal mean per layer
before averaging them, so the ranking does not depend on how many of each kind of token
you happened to calibrate on:

```python
st = sal_text.double()  / cnt_text.double().clamp(min=1)
si = sal_image.double() / cnt_image.double().clamp(min=1)
st = st / st.mean(dim=1, keepdim=True).clamp(min=1e-30)
si = si / si.mean(dim=1, keepdim=True).clamp(min=1e-30)
score = 0.5 * st + 0.5 * si
```

**This is a trade, not a free repair.** Both rankings were measured on both modalities at
256 kept:

| 256 kept, ranked by | text ppl | vs unpruned | caption ppl | vs unpruned |
|---|---:|---:|---:|---:|
| mean saliency | **3.8557** | **+5.5%** | 30.959 | +27.4% |
| **balanced (this repo)** | 4.1723 | +14.1% | **26.479** | **+8.9%** |

Balanced ranking buys back 14.5% of caption perplexity by giving up 8.2% of text
perplexity — it keeps experts that earn their place on images and little else. Neither
column dominates, so **pick by workload**:

- **Multimodal, or vision matters at all → this repo.** A +27% caption regression is not
  something you want shipped silently.
- **Text-only at 256 experts → prune with `--mode mean` instead** (the recipe repo does
  this in one flag); you will get 3.8557 and a vision path you should not use.
- **Anything that can load 272 experts → use
  [272E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E)**, which needs
  no such choice: +4.1% text and no measurable caption regression. It is strictly the
  better checkpoint, and vLLM simply will not load it.

21.4% of total saliency mass is dropped by this keep-set.

## The underlying reason: this model routes vision separately

The model carries **two** router biases — `gate.bias` and `gate.bias_vl` — and their
correlation is **−0.05** (min −0.45 across layers). Expert specialisation here is
modality-dependent.

Rank by text saliency alone and you discard **267 experts** that a combined ranking
keeps, including layer 0 expert 235: **376th of 384 on text, 21st on images**. At 272
kept, the resulting checkpoint is **worse than randomly pruning the same number of
experts** on vision:

| 272 kept, ranked by | caption ppl | vs unpruned |
|---|---:|---:|
| text + image saliency | 24.769 | +1.9% |
| random | 26.066 | +7.2% |
| text-only saliency | 28.173 | +15.9% |

…and its *text* perplexity looks fine — marginally better than the combined ranking
(3.7300 vs 3.7359 at 304 kept). Only an image-conditioned evaluation catches it.

If you prune this model yourself, calibrate on both modalities, and normalise between
them.

## Measurements

Held-out text perplexity, 65,504 tokens, verified zero overlap with the calibration
corpus:

| experts kept | perplexity | vs unpruned |
|---:|---:|---:|
| 384 (unpruned) | 3.6552 | — |
| 272 | 3.8060 | +4.13% |
| 256, mean-saliency ranking | 3.8557 | +5.49% |
| **256, balanced ranking (this repo)** | **4.1723** | **+14.14%** |
| 240 | 3.9156 | +7.12% |
| 192 | 4.1910 | +14.66% |

Backbone parameters fall from ~552B to ~371B. **Active parameters are unchanged** (6
routed + 1 shared per token) — pruning buys memory, not decode speed.

No downstream task benchmarks were run. Perplexity is not a benchmark.

## Serving

Verified on **2× GB10 (sm_121, 120 GB unified each)** under vLLM TP=2 over RoCE, with the
Engram tables read from NVMe rather than resident.

📄 **Full recipe, launcher and troubleshooting:
[github.com/Libertai/dsv41-flash-vllm-2x-spark](https://github.com/Libertai/dsv41-flash-vllm-2x-spark)**

### Throughput is workload-dependent, by a factor of three

Single stream, 600 tokens per case, DSpark k=5, greedy drafting, eager, 32K context:

| workload | decode c=1 | mean acceptance length (max 6) |
|---|---:|---:|
| counting ("1 to 220") | 49.4 tok/s | 5.90–6.00 |
| code generation | **30.4 tok/s** | 3.26–4.51 |
| reasoning | 28.3 tok/s | 3.13–3.81 |
| chat / explanation | 27.0 tok/s | 3.23–3.59 |
| prose (essay) | **19.2 tok/s** | 2.24–2.49 |

⚠️ **Do not quote a single number for this model, and never benchmark it on counting
prompts** — a degenerate sequence lets the drafter accept every token and reads ~2.6× the
honest rate. An earlier version of this card listed **13.7 tok/s**; that figure was measured
without speculative decoding on a prose-like prompt, i.e. close to the worst case.

| | |
|---|---|
| KV cache | **423,479 tokens** @ 32K ctx — 12.9× concurrency |
| prefill | 700–2,200 tok/s depending on length |
| weights | ~94 GiB/rank; Engram 47.2 GiB/rank left on disk |
| `--gpu-memory-utilization` | **0.90** (0.86 leaves no room for KV; above ~0.91 the free-memory check trips) |

### Context: the checkpoint says 1M, two GB10s serve 32K

vLLM will start at `--max-model-len 204800` and report a 1.27M-token KV pool. **It cannot
serve it.** Weights are 77% of the unified pool, so long-prefill working memory has nowhere
to go: a ~62K prompt at 64K ctx drops the host to 68 MB free and the engine is killed; a
~150K prompt made both boxes unreachable. Smaller prefill chunks do not help — the growth is
in the sparse-MLA indexer structures, not the chunk activations. **32K leaves a comfortable
~913 MB floor under load.** Four GPUs, or a smaller footprint, are what buy more.

### Five things this needs that are not obvious

1. **`--enforce-eager` is mandatory.** CUDA graphs fail here in both `PIECEWISE` and
   `FULL_AND_PIECEWISE` modes with `CUDA error: an illegal memory access was encountered`,
   consistent with the sm_120 sparse-MLA kernel receiving an uninstantiated batch shape.
   Reducing the capture set does not help. (A NaN-output failure mode was also seen on an
   earlier build — probe a new lane with `logprobs` before trusting its throughput.)
   Consequently **`enable_adaptive_verification` is unavailable**: it requires full graphs.
2. **`num_speculative_tokens` must be a MULTIPLE of `dspark_block_size` (5)** — k=7 is
   rejected at startup. And **k=5 is the fastest**: k=10 is 24–38% slower on every workload,
   because verifying k+1 tokens scales expert-weight traffic on a sparse MoE while
   per-draft acceptance falls.
3. **Turn on BOTH JIT warmups** (`enable_jit_warmup` *and* `enable_cutedsl_warmup`) and set
   `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`. Otherwise Triton and CuTeDSL kernels compile
   during inference and the engine dies mid-workload with
   `TimeoutError: RPC call to sample_tokens timed out`. Some CuTeDSL kernels are reached
   only on the long-prefill path, so short smoke tests will not expose this.
4. **FlashInfer has no sparse-MLA kernel for this model's index widths.** DSV4.1 mixes
   compress ratios, giving topk **1152** and **640**, and the instantiation list is keyed
   on `(num_heads, topk)` — so the entries you need depend on your TP degree
   (TP=2 → 32 heads, TP=4 → 16). Without them the engine dies at profiling with
   *"SM120 sparse-MLA has no decode kernel for this shape"*. **Patching the `.cu` alone does
   nothing** — the wheel ships a prebuilt AOT `sparse_mla_sm120.so` and `is_aot`
   short-circuits the JIT path. Mask the AOT directory, prebuild into a persistent cache,
   and mount that cache at serve time.
5. **Wire the reasoning parser, or the chain-of-thought lands in `content`.** Thinking is
   on by default and the chat template supplies the opening `<think>`, so with no parser
   you get raw reasoning followed by a bare `</think>` inside `content`, and a `reasoning`
   field sitting `null`. The parser ships and is registered as **`deepseek_v41`** — the
   same name serves both kinds:
   `--reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41`.
   The field is **`reasoning`**, not `reasoning_content`.
   ⚠️ With thinking on, budget **`max_tokens` ≥ ~3000** — a simple prompt spends ~2,800
   tokens (8.5k chars) reasoning, and a tighter cap returns empty `content`.

## Contents — read this before downloading

This repo carries **shards 1–46 only** (196.4 GiB): the pruned weights, the config, the
tokenizer, and the checkpoint's bundled reference implementation.

**Shards 47–48 are the two Engram tables and are NOT in this repo.** Pruning does not
touch them — they are byte-identical to the base model, 189.1 GiB, and re-uploading them
would cost every user a second copy of data they may already have. Fetch them from the
base model and drop them in:

```bash
hf download LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E --local-dir dsv41-reap256
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir dsv41-reap256 \
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
                "calibration": "262,144 text tokens + 256 images, saliency combined",
                "ranking": "balanced"}
```

## Limitations

- Calibration is 262,144 text tokens and 256 images. Larger would be better.
- Perplexity only; no task evaluations.
- The caption evaluation is 2,048 tokens. It resolves the cliff (+27%) and the
  random/text-only gaps comfortably; it does not resolve differences of a few percent.
- MTP layers are retained and were **not** pruned (they carry their own 128-expert pool,
  which the 384-expert keep-set does not index).

## License

Model: inherits the base model's license. Tooling: MIT.
