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

The pack in this repo is our [EXL3](https://github.com/turboderp-org/exllamav3) quantization of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, made with exllamav3's self-calibrated (SC) recipe. It is sized for 128 GB
Macs that stream the routed experts from SSD. The files here are exllamav3's own, and on Apple silicon mlx-serve serves
the streaming repack of this pack,
[OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3, branch `2.81bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw),
with the turboexl3_ds41 plugin.

As our recommended DeepSeek-V4.1 pack, it supersedes OpensourceWTF's earlier packs: the 2.9 bpw and 3.0 bpw streaming
repacks and [DeepSeek-V4.1-Flash-MTPLX-streaming-q2](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2).
The 2.9 bpw and 3.0 bpw packs stay available as branches of the two streaming repos: the
[`2.9bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.9bpw) branch of
OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3 and the
[`3.0bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3/tree/3.0bpw) branch of
OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3.

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

The pack is 380 GiB on disk, Engram included. As the table shows, the quantized non-expert layers are K6 with the MUL1
codebook, a few tensors of the vision tower and aligner are kept in BF16, and the router, norms, hyper-connections,
sinks and embeddings are as released. The DSpark drafter's routed experts are K4, as [DSpark drafter](#dspark-drafter)
explains.

### Expert usage profile

`expert-usage-v1.json` counts how often each of the 384 routed experts in each of the 40 layers was routed while this
pack served a small set of prompts. mlx-serve's turboexl3_ds41 plugin reads it from the root of the pack directory it
serves, with no setting needed, and fills each layer's empty expert slots with that layer's most used experts before the
first request. The profile changes which experts start resident, not the outputs, because it only seeds residency.

The profile was recorded on this pack's 40 x 384 routed experts (model type `deepseek_v41`), served by mlx-serve's
turboexl3_ds41 on an M5 Max, over 5 requests with 40,389 prompt and 4,104 decode tokens. The plugin refuses a profile
whose model type, layer count or expert count differs.

The corpus is the first ~60 KB of Python's `asyncio/base_events.py` with an explain task and a pytest task, the first
~60 KB of Zig's `std/hash_map.zig` with a review task, and two chat prompts, one on B-tree insertion in Go and one on a
three-day Kyoto plan. It has no benchmark text in it: none of its prompts shares a 64-character run with the code
prompts used to measure this pack's speed.

To turn the profile off, delete the file, set `MLX_SERVE_DSV41_RESIDENT_SEED=none`, or set
`MLX_SERVE_DSV41_EXPERT_USAGE_DIR` to another directory, one without the file or one holding your own profile.

## How to download and run

Download the pack with:

```sh
hf download OpensourceWTF/DeepSeek-V4.1-Flash-1017-exl3-2.81bpw --local-dir DeepSeek-V4.1-Flash-1017-exl3-2.81bpw
```

exllamav3 supports DeepSeek-V4.1 on its `dev` branch, at commit `ccc4c21` or later, and the pack was converted there.

On Apple silicon, mlx-serve serves the streaming repack,
[OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3, branch `2.81bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw),
with the turboexl3_ds41 plugin.

## Quality

| | PPL | KLD mean | KLD median | KLD p90 |
|---|---|---|---|---|
| release FP8/FP4 | 1.7730 | | | |
| noise floor | 1.7733 | 0.0023 | 0.0002 | 0.0050 |
| **this pack** | **1.8305** | **0.0396** | **0.0051** | **0.0917** |
| Mia-AiLab EXL3 EAQ v1 2.9 bpw, for comparison | 1.8306 | 0.0401 | 0.0058 | 0.0893 |

exllamav3's qbench compared the release's own FP8/FP4 weights, run in exllamav3, with the quantized pack. KLD is
KL(release ‖ pack) in nats over the 45,434 completion tokens of a held-out set of 20 conversations, with the full
129,280-token vocabulary. The noise floor is the release itself with BF16-scale rounding noise added after every layer.

The full data is on the streaming repo's card,
[OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3).
It covers the test set and method, the comparison with Mia-AiLab's EAQ v1 2.9 bpw, KLD by the release's confidence in
its top token, and Mia-AiLab's own published figures.

## Speed

The DSpark drafter only proposes tokens, and the model verifies every one, so swapping the drafter's experts leaves the
model's own tensors and the [Quality](#quality) figures unchanged. What moves is speed and memory. These were measured on
an M5 Max (128 GB) serving the streaming repack with mlx-serve's turboexl3_ds41 plugin, on a 16K-token code prompt with
1,024 generated tokens, with the K6 and K4 stages run in A B B A order:

| | K6 stages | K4 stages |
|---|---|---|
| greedy decode, tok/s | 47.8, 47.5 | 49.1, 49.1 (+3.0%) |
| sampled decode (T 1.0, top-p 0.95, top-k 20), tok/s | 40.8, 40.6 | 40.7, 41.0 (+0.4%) |
| drafts accepted a round, sampled / greedy | 3.94 / 3.91 | 3.81 / 3.91 |
| stage experts resident | 10.21 GB | 6.81 GB |

On the measured 16K code prompt, the greedy text was the same with either drafter, but the sampled text differs.
Speculative sampling keeps the model's distribution, but other drafts use the random draws differently.

## How the pack was made

The SC optimizer chose the expert rates per layer for a 2.8 bpw target and reached 2.813 bpw. `config.json` and
`quantization_config.json` record the drafter's rates as `mtp_bits: 6`, for the stages' attention, shared experts and
heads, and `mtp_expert_bits: 4`, for the stages' routed experts. exllamav3 reads neither field when it loads the pack.
Instead, each tensor's rate comes from its own shape.

### DSpark drafter

The routed experts of the three DSpark (MTP) stages, `mtp.{0,1,2}.ffn.experts.*` (3 x 128 experts), are
[Mia-AiLab](https://huggingface.co/Mia-AiLab)'s 4-bit EXL3 tensors from
[Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
at revision `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`, byte for byte. Both source shards were checked against their
LFS sha256. The stages' attention, shared experts and heads stay this pack's own K6.

### Engram tables

exllamav3 has no trellis codec for the Engram tables, so `ngram_embedding.safetensors` holds the release's own
`layers.{1,14}.engram.embed.{weight,scale}` tensors (F8_E4M3 values, F8_E8M0 scales per 32 values). Their bytes match
the release's shards 47 and 48.

### Chat template

`chat_template.jinja` is mlx-serve's DeepSeek-V4.1 template, transcribed from the release's `encoding/encoding.py`
(text path) and pinned byte for byte against that encoder. The release itself ships the encoder as Python only. The
calibration and evaluation traces were rendered with this template, which takes two parameters:

| Parameter | Values | Default |
|---|---|---|
| `thinking_mode` | `"chat"` or `"thinking"` (`enable_thinking` is accepted too) | chat |
| `reasoning_effort` | `low`, `medium`, `high`, `max`, or 1-100 | `high` |

## Credits

DeepSeek made the model and released it under the MIT license (`LICENSE`). turboderp created exllamav3, the EXL3 format
and the SC quantization pipeline. Mia-AiLab (MiaAI Lab) made the 4-bit EXL3 DSpark stage experts, which come from
[DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw).

## License

DeepSeek released DeepSeek-V4.1-Flash-1017 under the MIT license, and the `LICENSE` file in this repo is that license.

## Changelog

| Date | Change |
|---|---|
| 2026-11-04 | The card was rewritten for readability. No figures changed. |
| 2026-11-03 | The Mia-AiLab comparison moved to the card of [OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3). |
| 2026-11-01 | Added `chat_template.jinja` (see [Chat template](#chat-template)). `config.json` and `quantization_config.json` gained `mtp_expert_bits: 4` for the DSpark stages' K4 routed experts, and the card gained the full comparison with Mia-AiLab's EAQ v1 2.9 bpw and Mia-AiLab's own published figures. No weights changed. |
| 2026-11-01 | Added `expert-usage-v1.json`, the expert usage profile that seeds mlx-serve's resident experts (see [Expert usage profile](#expert-usage-profile)). No weights changed. |
| 2026-11-01 | The DSpark stages' routed experts are now Mia-AiLab's K4 EXL3 tensors (they were this pack's K6). `model-00037-of-00038.safetensors`, `model-00038-of-00038.safetensors` and `model.safetensors.index.json` changed, and every other tensor is the same. |
