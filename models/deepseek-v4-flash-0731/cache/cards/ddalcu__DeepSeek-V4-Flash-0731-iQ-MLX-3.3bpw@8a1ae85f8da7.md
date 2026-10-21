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

# DeepSeek-V4-Flash-0731 — iQ-MLX 3.33 bpw

An **iQ-MLX** conversion of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) at
**3.33 bits per weight**, built to run the full 284B-A13B model on
a single Apple Silicon Mac with **128 GB or more** of unified memory. DSpark MTP heads are included, but wont fit in 128mb, only enable DSpark if you have over 128GB.

In order to fit this model useable on 128GB with proper context size, you will need to increase the MacOS GPU Allocation using `sudo sysctl iogpu.wired_limit_mb=126000` and quit all apps, all browsers.

## What is iQ-MLX

iQ-MLX brings llama.cpp's imatrix school of quantization to native MLX affine
layouts: per-input-channel importance is measured on real calibration traffic
(mean-squared activation over millions of tokens, per expert), every
quantization group's scale/bias is chosen by an importance-weighted
least-squares search instead of min/max, and the per-layer/per-projection bit
widths themselves are allocated by a greedy error-per-byte knapsack over the
measured error surface — the byte budget goes exactly where the calibration
says it buys the most quality. The packs stay byte-compatible with stock MLX
affine, so the result loads like any MLX model: no custom codec, no
dequantize-on-load, no runtime cost.

How it differs from what you may know:

- **vs plain MLX 4bit / mixed_2_6**: those are uncalibrated min/max with
  static recipes; iQ-MLX measures which channels and layers matter and fits
  scales to that.
- **vs DWQ**: DWQ distills against the fp teacher on a GPU (hours of training);
  iQ-MLX is a CPU-only search you can run on the same box that serves the
  model, at comparable calibration quality for MoE experts.
- **vs AWQ**: activation-aware scale folding is a min/max compensation. We
  measured it against the weighted search on this model: redundant on gate/up
  (under 1 percent) and harmful on down projections — so iQ-MLX deliberately
  does not fold.
- **vs GGUF IQ/UD**: same philosophy (imatrix calibration, dynamic per-layer
  bits), but emitting native MLX affine tensors instead of GGUF — no llama.cpp
  in the serving path.

The bpw figure in the name is the honest size metric: stored bits (weights +
scales + biases) divided by parameters, comparable across iQ-MLX builds and
against GGUF tiers (IQ2_XXS is 2.06, Q4_K_M is about 4.8). Higher bpw = less
quantized.

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
| Routed expert gate/up (`w1`/`w3`), early layers | affine **2-bit**, group size 128, **imatrix-calibrated** |
| Routed expert gate/up (`w1`/`w3`), layers 7-8 and 16-38 | affine **3-bit**, group size 128, **imatrix-calibrated** |
| Routed expert down (`w2`) | affine **3-bit**, group size 128, **imatrix-calibrated** |
| Routed experts, tail layers 39-42 (all projections) | affine **4-bit**, group size 64, **imatrix-calibrated** |
| Attention, shared experts, indexer, `main_proj` | affine **8-bit**, group size 64 |
| Embedding + LM head | affine **8-bit**, group size 64 |
| DSpark draft stages (`mtp.*`) — experts | affine **4-bit**, group size 64 |
| Compressor `wkv`/`wgate`, indexer `weights_proj`, router `gate.weight` | bf16 |
| Norms, hyper-connection params, `ape`, attention sinks, router bias, hash table | verbatim |

The layer/projection split is not hand-picked: the whole error surface (every
layer x projection at every candidate width, sampled per expert against the
imatrix objective) was measured, and the byte budget allocated greedily by
error reduction per byte. The data put nearly all of it into gate/up on the
later two thirds of the stack — consistent with what agent-level testing
showed about late layers and decision quality. The 4-bit tail (layers 39-42)
is kept from the previous build: it is what eliminated turn-level agent
repetition loops in A/B testing.

The routed experts — 277B of the 284B — are quantized with the iQ-MLX
activation-calibrated search rather than plain min/max: per-input-channel
importance comes from an importance matrix collected over **2.9M tokens** of
chat-formatted traffic through the 0731 weights themselves (per-expert channel
granularity), and each quantization group's scale/bias pair is chosen by a
weighted multi-start search with alternating least-squares refinement (the
llama.cpp `make_qkx2_quants` pattern). Channels that actually fire reconstruct
better; at 2-3 bits this is worth more than finer group granularity, which is
why the experts use group size 128 and spend the saved bytes on extra bits
where the error surface says they matter.

One more choice worth explaining: the DSpark draft stages keep **4-bit**,
uncalibrated (the imatrix does not cover them). They are a rounding error on
disk, and a draft the trunk rejects costs a full verify forward, so their
quality multiplies throughput.

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

- Apple Silicon Mac, **128 GB+** unified memory (~118 GB resident; raise the
  GPU limit with `sudo sysctl iogpu.wired_limit_mb=124000`). On 128 GB
  machines mlx-serve auto-disables DSpark when the stages don't fit and
  serves serial — `--dspark` engages on larger machines (192 GB+).
- macOS 26.2 or newer
- [mlx-serve](https://github.com/ddalcu/mlx-serve)

Built with `tests/convert_dsv4_weights.py` from the mlx-serve repo (the
iQ-MLX pipeline: imatrix parser, weighted affine search, and the greedy
bit-allocation sweep).
