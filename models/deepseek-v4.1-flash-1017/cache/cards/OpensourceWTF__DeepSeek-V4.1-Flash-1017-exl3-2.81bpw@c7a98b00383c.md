---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
tags:
  - exl3
  - exllamav3
  - deepseek
  - moe
---

# DeepSeek-V4.1-Flash-1017, EXL3 2.81 bpw

Streaming repack for mlx-serve: [OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw), branch `2.81bpw`.

An [EXL3](https://github.com/turboderp-org/exllamav3) quantization of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, made with exllamav3's self-calibrated (SC) recipe. It is sized for 128 GB
Macs that stream the routed experts from SSD.

It replaces OpensourceWTF's earlier DeepSeek-V4.1 packs:
[DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.9bpw](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.9bpw),
[DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw)
and [DeepSeek-V4.1-Flash-MTPLX-streaming-q2](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2).

These are exllamav3's own files. The streaming repack of this pack for mlx-serve is published as
[OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3, branch `2.81bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw).

## What is in it

| part | format | size |
|---|---|---|
| routed experts (40 layers x 384) | EXL3 K3; K2 on layers 23 and 32-39 | 189.2 GB |
| attention, compressor, indexer projection, shared expert, LM head | EXL3 K6 | 7.7 GB with the embeddings |
| MTP (DSpark) stages, 3 layers: attention, shared expert, heads | EXL3 K6 | 0.6 GB |
| MTP (DSpark) stages' routed experts, 3 x 128 | EXL3 K4, Mia-AiLab's (see below) | 6.8 GB |
| vision tower and aligner | EXL3 K6, a few tensors kept in BF16 | 0.9 GB |
| router, norms, hyper-connections, sinks, embeddings | as released | |
| Engram n-gram tables (layers 1 and 14) | the release's FP8 tables, byte for byte | 202.8 GB |

- **Expert rates:** the SC optimizer chose them per layer for a 2.8 bpw target and reached 2.813 bpw. Every non-expert
  tensor of the model is K6, and the codebook is MUL1. The DSpark drafter's routed experts are K4 (next section).
  `config.json` and `quantization_config.json` record this as `mtp_bits: 6` (the stages' attention, shared experts and
  heads) and `mtp_expert_bits: 4` (the stages' routed experts). exllamav3 reads neither field when it loads the pack;
  each tensor's rate comes from its own shape.
- **Chat template:** `chat_template.jinja` is mlx-serve's DeepSeek-V4.1 template, transcribed from the release's
  `encoding/encoding.py` (text path) and pinned byte for byte against that encoder. It takes `thinking_mode`
  (`"chat"` or `"thinking"`; `enable_thinking` is accepted too, and the default is chat) and `reasoning_effort`
  (`low`, `medium`, `high`, `max`, or 1-100; default `high`). The calibration and evaluation traces below were rendered
  with it. The release itself ships the encoder as Python only.
- **Engram tables:** exllamav3 has no trellis codec for them, so `ngram_embedding.safetensors` holds the release's own
  `layers.{1,14}.engram.embed.{weight,scale}` tensors (F8_E4M3 values, F8_E8M0 scales per 32 values). Their bytes match
  the release's shards 47 and 48.
- **Size:** 380 GiB on disk, Engram included.

## DSpark drafter

The three DSpark (MTP) stages' routed experts, `mtp.{0,1,2}.ffn.experts.*` (3 x 128 experts), are
[Mia-AiLab](https://huggingface.co/Mia-AiLab)'s 4-bit EXL3 tensors from
[Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
at revision `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`, byte for byte (both source shards checked against their LFS
sha256). The stages' attention, shared experts and heads stay this pack's own K6.

The drafter only proposes tokens and the model verifies every one, so the model's own tensors and the quality figures
below are unchanged. What moves is speed and memory, measured on an M5 Max (128 GB) serving the streaming repack with
mlx-serve's turboexl3_ds41 plugin (a 16K-token code prompt, 1,024 generated tokens, K6 and K4 stages in A B B A order):

| | K6 stages | K4 stages |
|---|---|---|
| greedy decode, tok/s | 47.8, 47.5 | 49.1, 49.1 (+3.0%) |
| sampled decode (T 1.0, top-p 0.95, top-k 20), tok/s | 40.8, 40.6 | 40.7, 41.0 (+0.4%) |
| drafts accepted a round, sampled / greedy | 3.94 / 3.91 | 3.81 / 3.91 |
| stage experts resident | 10.21 GB | 6.81 GB |

On the measured 16K code prompt the greedy text was the same with either drafter; the sampled text differs:
speculative sampling keeps the model's distribution, but other drafts use the random draws differently.

## Quality

exllamav3's qbench compared the release's own FP8/FP4 weights, run in exllamav3, with the quantized pack: KL(release
‖ pack) in nats over the 45,434 completion tokens of a held-out set of 20 conversations, with the full 129,280-token
vocabulary. The noise floor is the release itself with BF16-scale rounding noise added after every layer.

| | PPL | KLD mean | KLD median | KLD p90 |
|---|---|---|---|---|
| release FP8/FP4 | 1.7730 | | | |
| noise floor | 1.7733 | 0.0023 | 0.0002 | 0.0050 |
| **this pack** | **1.8305** | **0.0396** | **0.0051** | **0.0917** |
| Mia-AiLab EXL3 EAQ v1 2.9 bpw, for comparison | 1.8306 | 0.0401 | 0.0058 | 0.0893 |

The full data (the test set and method, the comparison with Mia-AiLab's EAQ v1 2.9 bpw, KLD by the release's
confidence in its top token, and Mia-AiLab's own published figures) is on the streaming repo's card,
[OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3).

## Running it

- **exllamav3:** DeepSeek-V4.1 support is on the `dev` branch (commit `ccc4c21` or later); the pack was converted there.
- **Apple Silicon:** mlx-serve serves the streaming repack,
  [OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3, branch `2.81bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw),
  with the turboexl3_ds41 plugin.

## Expert usage profile

`expert-usage-v1.json` counts how often each of the 384 routed experts in each of the 40 layers was routed while this
pack served a small set of prompts. mlx-serve's turboexl3_ds41 plugin reads it from the root of the pack directory it
serves, with no setting needed, and fills each layer's empty expert slots with that layer's most used experts before the
first request.

- **What it changes:** which experts start resident, not the outputs. It only seeds residency.
- **Recorded on:** this pack's 40 x 384 routed experts (model type `deepseek_v41`), served by mlx-serve's turboexl3_ds41
  on an M5 Max: 5 requests, 40,389 prompt and 4,104 decode tokens. The plugin refuses a profile whose model type, layer
  count or expert count differs.
- **Corpus:** the first ~60 KB of Python's `asyncio/base_events.py` with an explain task and a pytest task, the first
  ~60 KB of Zig's `std/hash_map.zig` with a review task, and two chat prompts (B-tree insertion in Go, a three-day
  Kyoto plan). It has no benchmark text in it: none of its prompts shares a 64-character run with the code prompts used
  to measure this pack's speed.
- **To turn it off:** delete the file, set `MLX_SERVE_DSV41_RESIDENT_SEED=none`, or set
  `MLX_SERVE_DSV41_EXPERT_USAGE_DIR` to another directory (one without the file, or one holding your own profile).

## Credits

- **DeepSeek:** the model, released under the MIT license (`LICENSE`).
- **turboderp:** exllamav3, the EXL3 format and the SC quantization pipeline.
- **Mia-AiLab (MiaAI Lab):** the 4-bit EXL3 DSpark stage experts, from
  [DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw).

## Changelog

- 2026-11-01: added `chat_template.jinja` (see What is in it); `config.json` and `quantization_config.json` gained
  `mtp_expert_bits: 4` for the DSpark stages' K4 routed experts; the card gained the full comparison with Mia-AiLab's
  EAQ v1 2.9 bpw and Mia-AiLab's own published figures. No weights changed.
- 2026-11-01: added `expert-usage-v1.json`, the expert usage profile that seeds mlx-serve's resident experts (see
  Expert usage profile). No weights changed.
- 2026-11-01: the DSpark stages' routed experts are now Mia-AiLab's K4 EXL3 tensors (they were this pack's K6).
  `model-00037-of-00038.safetensors`, `model-00038-of-00038.safetensors` and `model.safetensors.index.json` changed;
  every other tensor is the same.
