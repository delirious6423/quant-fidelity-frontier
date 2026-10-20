---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - deepseek
  - moe
  - mixed-precision
  - apple-silicon
  - rapid-mlx
---

# DeepSeek-V4.1-Flash-1017 — Rapid-MLX mixed-precision (256 GB Macs)

A mixed-precision MLX build of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
sized to run on a single **256 GB Apple Silicon Mac** with
[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).

- **No experts removed.** All 40 × 384 routed experts are kept.
- **Bits go where they were measured to matter.** The routed experts hold 96% of
  the parameters. They get 2/3/4-bit widths per layer and per projection. Every
  other weight stays at high precision.
- **~206.5 GB resident** (weights the model keeps in RAM), plus 122.9 GB of
  Engram tables that stay on SSD. The download is 329 GB in total.

## Quick start

> Requires Rapid-MLX 0.16.5 or newer.

```bash
pip install -U rapid-mlx
rapid-mlx serve deepseek-v41-flash-1017
```

This starts an OpenAI-compatible server on `http://127.0.0.1:8000/v1`. The
first run downloads the pinned revision and verifies its file hashes.

## Requirements and current limits

| Item | Value |
|---|---|
| Unified memory | 256 GB Mac (the server refuses to start below the guard) |
| Free disk | ~330 GB (Engram tables are read from SSD on demand) |
| Decoding | greedy only (`temperature` 0); `seed` is rejected |
| Input | text only, ≤ 8,192 prompt tokens |
| Output | ≤ 4,096 tokens per request |
| Tools / function calling | not yet supported by this serving lane |
| Speculative decoding | built-in DSpark draft (`dspark/`, 4-bit experts) |

Sampling, tool calling, and longer inputs and outputs are planned follow-ups for the serving
lane. They are not limits of the weights.

## What is in this build

| Component | Precision |
|---|---|
| Routed experts, layers 0–14, 16–19, 32–34 | gate/up/down 2/2/2-bit |
| Routed experts, layer 15 | 3/3/3-bit |
| Routed experts, layers 20–31 | 2/2/3-bit (down_proj 3-bit) |
| Routed experts, layers 35–38 | 3/3/3-bit |
| Routed experts, layer 39 | 4/4/3-bit |
| Attention, shared experts, indexer/compressor | 8-bit |
| Embeddings, LM head, router | BF16 |
| Engram (layers 1, 14) | 4-bit, group 32, offloaded to SSD |
| DSpark draft head | 8-bit dense, 4-bit experts |

### How the widths were chosen

1. **Reference.** The source FP4/FP8 checkpoint was streamed one layer at a time
   to produce teacher logits on 38 held-out text slices. These cover chat, code,
   math, tool-call transcripts, agentic terminal sessions, law, medicine,
   finance, science, and 20 languages including low-resource ones. A 2-bit
   teacher was not used.
2. **Pruning was evaluated and rejected for this size.** Contrastive hot/cold
   expert scoring on document-disjoint calibration halves gave a very stable
   "important" set (Jaccard 0.98), but unstable "unimportant" candidates
   (Jaccard 0.26–0.35). Removing experts on that evidence was not safe, so none
   are removed.
3. **Measured layer sensitivity.** Mixed-bit variants were built and scored by
   token-level KL against the reference. The late layers (35–39) and the
   middle-to-late band (20–34) dominate quality. Layers 0–14 barely matter.
   Within the middle band, `down_proj` alone recovers most of the gain at a third
   of the memory.

## Quality

Teacher-forced next-token KL divergence vs. the source-precision reference.
There are 38 held-out slices of 512 tokens each; lower is better.

| Build | Resident | Mean KL (nats) | Worst slice | Mean top-1 flip |
|---|---|---|---|---|
| Uniform 2-bit experts | 180 GB | 2.02 | 5.93 | 34% |
| **This build** | **206.5 GB** | **0.439** | **0.749** | **21%** |

Per-domain KL for this build:

| Domain | KL |
|---|---|
| code | 0.128 |
| agentic terminal | 0.156 |
| math | 0.179 |
| tool-call transcripts | 0.187 |
| general chat (EN) | 0.251 |
| general chat (ZH) | 0.246 |
| hardest slice (Chinese folk-customs text) | 0.749 |

Non-English and long-tail knowledge text drifts more than code, math and
agentic text.

### Task benchmarks

Greedy, non-thinking mode, fixed sample IDs, through the Rapid-MLX server.

| Benchmark | Items | Score | Notes |
|---|---|---|---|
| GSM8K (test) | 250 | **91.2%** | exact numeric match, zero-shot chat, 2,048-token cap |
| HumanEval | 164 (all) | **89.6%** pass@1 | executed in a no-network sandbox |
| MMLU-Pro | 300 (stratified) | **66.7%** | zero-shot letter answer; 26 items hit the 512-token cap |

These are small, fixed subsets without thinking mode (512-token cap unless
noted). They are not leaderboard protocols, so compare them across builds, not
against published numbers.

### Against the original model (early, sampled)

A first, deliberately small check on reasoning benchmarks that the original
model card also reports. Thinking mode, greedy, fixed random samples. Because
this serving lane currently caps output at 4,096 tokens (the original model is
evaluated with ≥ 256K), answers whose reasoning was cut off were finished by
asking for the final answer from the reasoning so far.

| Benchmark | Sample | This build | Original | Notes |
|---|---|---|---|---|
| GPQA Diamond | 15 of 198 | 11/15 (73%) | 90.9 (V4.1-Flash) | 7 of 15 hit the 4,096-token cap |
| GPQA Diamond, reasoning completed | 8 | 7/8 (88%) | 90.9 (V4.1-Flash) | |
| HLE, text-only | 30 of 2,158 | 3/30 (10%) | 44.0 | 20 of 30 hit the cap; not a quality comparison |

Read these as direction, not as scores: the samples are small (± 10 points or
more) and the output cap dominates the HLE result. Where the model finishes
its reasoning, GPQA accuracy is within a few points of the original. Full,
protocol-matched runs will follow once the serving lane supports long outputs.

### Speed and memory (Mac Studio M3 Ultra, 256 GB)

| Prompt | Decode (tok/s) | Prefill (tok/s) |
|---|---|---|
| 1k tokens | 19.9 | ~167 |
| 8k tokens | 18.8 | ~170 |

Greedy decoding with the built-in DSpark draft. Prefill is a client-side estimate.

**Memory.** The weights are loaded into RAM up front: ~206.5 GB resident, plus
working buffers. The measured peak process footprint is 228 GB (8k-token prompt,
otherwise idle Mac Studio). On a 256 GB Mac, quit other memory-heavy apps
before starting the server. If macOS runs out of memory and swap during loading,
it terminates the server.

## Reproducing

The build and evaluation tooling lives in the Rapid-MLX repository:

- `scripts/deepseek_v41_rd_build.py` / `scripts/repack_deepseek_v41_native.py`
- `scripts/evaluate_deepseek_v41_rd.py`
- `scripts/eval_deepseek_v41_tasks.py`

The per-layer bit allocation and byte breakdown ship as `rapid-mlx-allocation.json`.

## License

MIT, same as the base model. See `LICENSE`.
