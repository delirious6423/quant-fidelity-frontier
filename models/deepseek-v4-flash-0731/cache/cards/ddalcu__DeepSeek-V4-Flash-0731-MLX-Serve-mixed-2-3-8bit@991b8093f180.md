---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: mlx-serve
tags:
  - mlx
  - mlx-serve
  - deepseek
  - quantized
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 — imatrix-calibrated mixed 2/3/4/8-bit for mlx-serve

## Please use https://huggingface.co/ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw Instead of this model

A 115.4 GB mixed-precision MLX conversion of
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), built to run the full 284B-A13B model
on a single Apple Silicon Mac with **128 GB or more** of unified memory.

Runs on [**mlx-serve**](https://github.com/ddalcu/mlx-serve) — a native Zig
inference server for MLX models on Apple Silicon, with no Python in the serving
path. It speaks the OpenAI, Anthropic and Ollama HTTP APIs, so existing clients
(Claude Code, pi, opencode, Open WebUI, …) point at it unchanged.

```bash
mlx-serve --model <this-repo-dir> --serve --port 11434
```

DeepSeek-V4-Flash's architecture is implemented natively in mlx-serve: MQA over
a single 512-dim latent, window-128 raw attention plus gated-pooling compressed
history with a top-512 indexer, per-head attention sinks, Sinkhorn hyper-
connections, hash-routed early MoE layers, and the DSML tool-call format. No
llama.cpp, no GGUF conversion, no Python runtime.

## How it was quantized

Every tensor class is sized by what it costs and how much it matters, rather
than one global bit width:

| Tensors | Precision |
|---|---|
| Routed expert gate/up (`w1`/`w3`), layers 0-38 | affine **2-bit**, group size 128, **imatrix-calibrated** |
| Routed expert down (`w2`), layers 0-38 | affine **3-bit**, group size 128, **imatrix-calibrated** |
| Routed experts, layers 39-42 (`w1`/`w2`/`w3`) | affine **4-bit**, group size 64 |
| Attention, shared experts, indexer, `main_proj` | affine **8-bit**, group size 64 |
| Embedding + LM head | affine **8-bit**, group size 64 |
| DSpark draft stages (`mtp.*`) — experts | affine **4-bit**, group size 64 |
| Compressor `wkv`/`wgate`, indexer `weights_proj`, router `gate.weight` | bf16 |
| Norms, hyper-connection params, `ape`, attention sinks, router bias, hash table | verbatim |

The routed experts — 277B of the 284B — are quantized with an
activation-calibrated search rather than plain min/max: per-input-channel
importance comes from an importance matrix collected over **1.5M tokens** of
chat-formatted text (antirez's DeepSeek-V4-Flash imatrix from the ds4
gguf-tools, per-expert channel granularity), and each quantization group's
scale/bias pair is chosen by a weighted multi-start search with alternating
least-squares refinement (the llama.cpp `make_qkx2_quants` pattern). Channels
that actually fire reconstruct better; at 2-3 bits this is worth more than
finer group granularity, which is why the experts use group size 128 and spend
the saved bytes nowhere — the model just gets smaller and faster.

The last four layers keep their experts at 4-bit. This is not a nicety, it is
what makes the model usable as a coding agent: with 2-bit experts all the way
to the top, the final layers sit on near-tie logit boundaries and the model
falls into turn-level repetition loops in agent sessions, re-issuing the same
tool call dozens of times instead of concluding. Raising just those four
layers fixes it (details in the revision history below).

Two more choices worth explaining. The down-projection keeps **3-bit** while
gate/up drop to 2-bit: it is the most quantization-sensitive of the three. The
DSpark draft stages keep **4-bit**, uncalibrated (the imatrix does not cover
them): they are a rounding error on disk, and a draft the trunk rejects costs
a full verify forward, so their quality multiplies throughput.

The compressor path is fp32-sensitive by design and the router is read raw, so
neither is quantized. Lookup tables (embeddings, the token→expert hash, DSpark's
Markov table) are never packed — they are gathered, not multiplied.

Conversion is exact where it can be: the source's fp8 (e4m3 + e8m0 block scales)
and fp4 (e2m1 + e8m0 group scales) formats all fit losslessly in bf16, so the
weights are decoded exactly before requantization. The calibrated expert packs
are byte-compatible with MLX's affine layout, so the mirror is engine-native —
no dequantize-on-load step at runtime.

## What is included

Weights, tokenizer, and a chat template transcribed from the release's own
`encoding/encoding_dsv4.py` and verified **byte-exact** against it across chat
and thinking modes, tool definitions, DSML tool-call history, multi-turn
drop-thinking, and all three reasoning-effort levels. `generation_config.json`
carries the reference's own default sampling (temperature 0.6), not the wild
1.0/1.0 signature the source ships.

DSpark speculative-decoding weights (3 draft stages) are included and
converted; mlx-serve drives them with `--dspark` (block-parallel speculative
decode, greedy and sampled).

## Requirements

- Apple Silicon Mac, **128 GB+** unified memory (~98 GB resident, +11 GB with `--dspark`)
- macOS 26.2 or newer
- [mlx-serve](https://github.com/ddalcu/mlx-serve)

Built with `tests/convert_dsv4_weights.py` from the mlx-serve repo.

## Revision history

- 2026-08-02 (tail fix): routed experts on the last 4 layers (39-42) raised
  from 2/3-bit to 4-bit, group size 64. The uniform low-bit build had a
  behavior bug in agent use: it would finish the work, tests green, then loop
  on the same verification tool call until the client gave up. In a repeated
  A/B on an identical coding-agent task the uniform build looped in 3 of 4
  runs and spent 16-28K tokens per task; this build looped in 0 of 3 and
  spent 5-8K tokens, with identical correctness on the task's 50-vector
  check and identical decode speed (~53 tok/s with `--dspark` on an M-series
  128 GB). 115.4 GB instead of 109.2. Every shard below layer 39 is
  byte-identical to the previous revision.
- 2026-08-02: routed experts requantized with a fresh imatrix collected on the
  0731 weights themselves (2.9M tokens of the chat-v2 corpus through antirez's
  official 0731 GGUF, 747M routed-expert observations). The previous revision
  was calibrated with the only published imatrix at the time, which turned out
  to be collected on the preview checkpoint; the 0731 retrain moved the
  per-expert activation statistics substantially (median correlation 0.66
  against preview), mostly on the down projections. Better verbatim precision
  in code and text. Spine, embeddings and DSpark draft stages are
  byte-identical to the previous revision.
- 2026-08-01: routed experts rebuilt with imatrix-calibrated quantization at
  group size 128 (previously uncalibrated min/max at group size 64). Better
  character-level fidelity in verbatim text and code edits, 109.2 GB instead
  of 117.8, and more usable context headroom on 128 GB Macs.
- Earlier revisions remain available through this repo's git history.
