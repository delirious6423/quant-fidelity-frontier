---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: mlx-serve
pipeline_tag: text-generation
tags:
  - deepseek_v41
  - exl3
  - mlx-serve
  - streaming
  - moe
---

# DeepSeek-V4.1-Flash-1017, streaming EXL3 packs for mlx-serve

This repo holds streaming [EXL3](https://github.com/turboderp-org/exllamav3) packs of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, for [mlx-serve](https://github.com/ddalcu/mlx-serve). Each pack is a
branch. In every pack the trunk loads once, and the routed experts and the Engram tables are read from SSD as they are
needed, so a Mac with 128 GB can serve the model. `main` holds this card, its charts (`assets/`) and the license.
The turboexl3_ds41 plugin, which serves every branch but `2.9bpw`, needs an M5-class GPU.

`2.81bpw` is our pack and the one we recommend. `1.68bpw`, `2.25bpw` and `2.68bpw` are turboderp's quants,
served by the same plugin with its fractional-layer support. `2.9bpw` is Mia-AiLab's quant on the older mlx-stream plugin.

## Branches

| Branch | Routed experts | Trunk | Size | Source and credit | Served by | Sweep |
|---|---|---|---:|---|---|---|
| [`2.81bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw) | EXL3 SC 2.81 bpw: K3, K2 on layers 23 and 32-39 | K6 | 408.2 GB | Our quant, made with turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3) SC pipeline and re-laid for streaming from its raw exllamav3 form; the DSpark stage experts are [Mia-AiLab](https://huggingface.co/Mia-AiLab)'s K4 | turboexl3_ds41 ([PR #35](https://github.com/CommunityServe/plugins/pull/35)) | 1K–256K |
| [`1.68bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/1.68bpw) | EXL3 SC 1.68 bpw: K1.5, K2 on layers 0-5 and 35-39 | K5, head K6 | 326.7 GB | turboderp's quant, his [`1.68bpw`](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/1.68bpw) branch (revision `4a72d559`) | turboexl3_ds41 with fractional layers ([PR #35](https://github.com/CommunityServe/plugins/pull/35)) | 1K–256K |
| [`2.25bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.25bpw) | EXL3 SC 2.25 bpw: K2, K2.5 on layers 0-5 and 35-39 | K6 | 361.5 GB | turboderp's quant, his [`2.25bpw`](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/2.25bpw) branch (revision `9183eaae`) | turboexl3_ds41 with fractional layers ([PR #35](https://github.com/CommunityServe/plugins/pull/35)) | 1K–256K |
| [`2.68bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.68bpw) | EXL3 SC 2.68 bpw: K2.5, K3 on layers 0-5 and 35-39 | K6 | 395.4 GB | turboderp's quant, his [`2.68bpw`](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/2.68bpw) branch (revision `608c237e`) | turboexl3_ds41 with fractional layers ([PR #35](https://github.com/CommunityServe/plugins/pull/35)) | 1K–256K |
| [`2.9bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.9bpw) | EXL3 EAQ v1 2.9 bpw: K3, K2 on layers 18-22 | the release's values, as exact MLX repacks | 417.8 GB | Mia-AiLab's quant, [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) (revision `45ee5d0f`) | [mlx-stream](https://github.com/davidtai/mlx-stream) | not in the sweep |

Sizes are in GB, where 1 GB is 10^9 bytes. K is the bits per weight of a layer's routed experts. Each branch's card
lists its files and has its own sweep rows.

### Sources

| Repo | What it holds |
|---|---|
| [turboderp/DeepSeek-V4.1-Flash-1017-exl3](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) | turboderp's EXL3 SC quants for ExLlamaV3, the source of `1.68bpw`, `2.25bpw` and `2.68bpw`, with his KLD and perplexity charts and his AIME-25 and GPQA-Diamond scores. |
| [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) | Mia-AiLab's EAQ v1 2.9 bpw quant, the source of `2.9bpw` and of the DSpark stage experts in `2.81bpw`. |

### Deprecated repos

These earlier OpensourceWTF repos are deprecated stubs that hold no files; their files remain in each repo's history.
This repo replaces them.

- [OpensourceWTF/DeepSeek-V4.1-Flash-1017-exl3-2.81bpw](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-exl3-2.81bpw): held the raw
  exllamav3 quant that `2.81bpw` is repacked from.
- [OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.9bpw](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.9bpw):
  its files moved to this repo's `2.9bpw` branch.
- [OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3):
  held DeepSeek-V4.1-Flash before the 1017 refresh, 3.0 bpw, for mlx-stream.
- [OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2): held the same
  earlier base with 2-bit MLX affine experts, for MTPLX.

## How to download and run

To download one branch, pass its name to `--revision`. For example, this command downloads `2.81bpw`:

```sh
hf download OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3 --revision 2.81bpw --local-dir DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.81bpw
```

**`2.81bpw`** runs on mlx-serve 26.11.1 or later with the turboexl3_ds41 plugin, [plugins/turboexl3_ds41](https://github.com/CommunityServe/plugins/tree/main/plugins/turboexl3_ds41) on
CommunityServe/plugins main, merged in [CommunityServe/plugins#35](https://github.com/CommunityServe/plugins/pull/35). The plugin needs an M5-class GPU and
refuses any other at load. Build and install it with the plugins repo's installer:

```sh
git clone https://github.com/CommunityServe/plugins && cd plugins
scripts/install.sh turboexl3_ds41
```

Turn plugins on (`"enabled": true` in `~/.mlx-serve/plugins.json`) and start the server on the branch:

```sh
mlx-serve --serve --model DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.81bpw --port 8080
```

The log line `[plugin] turboexl3_ds41 serves deepseek_v41` shows the plugin took the pack.
[How the sweep was measured](#how-the-sweep-was-measured) lists the settings behind the published figures.

The published figures ran with the three host additions of [ddalcu/mlx-serve#1065](https://github.com/ddalcu/mlx-serve/pull/1065), merged as `e5cdd78`: `HostGpu.alias` (the plugin's
bank pool), `HostGpu.command_buffer_limits` (MLX's command-buffer split, raised for prompts) and keyed sampling.
mlx-serve 26.11.2, or main from `e5cdd78`, has them; 26.11.1 does not. To serve with the published settings, use
26.11.2 with `MLX_SERVE_KEYED_SAMPLING=1`. On 26.11.1 the plugin still loads, with its bank pool off, MLX's own
command-buffer split and the host's default sampler; in the
[parity check](#parity-of-the-plugin-port), a server without the additions measured prefill 2.5% to 2.8% lower on
`2.81bpw`.

**`1.68bpw`, `2.25bpw` and `2.68bpw`** use exllamav3's half-integer rates, 1.5 and 2.5 bits per weight, on some routed
layers. The plugin serves those layers only with its fractional-layer support (`MLX_SERVE_DSV41_FRAC`), which is in the
same plugin, merged in the same pull request.

**`2.9bpw`** runs on the [mlx-stream](https://github.com/davidtai/mlx-stream) plugin. Its K2 layers need mlx-stream
[PR #25](https://github.com/davidtai/mlx-stream/pull/25) or later, merged on 2026-10-21. The
[`2.9bpw` branch card](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/blob/2.9bpw/README.md)
has the command that starts the server.

## Quality

| | `2.81bpw` | `2.9bpw` |
|---|---:|---:|
| size | 408.2 GB | 417.8 GB |
| main model weights, as qbench counts them | 182.0 GiB, 2.837 bpw | 187.1 GiB, 2.916 bpw |
| PPL (release FP8/FP4: 1.7730; noise floor: 1.7733) | 1.8305 | 1.8306 |
| KLD mean (noise floor: 0.0023) | 0.0396 | 0.0401 |
| KLD median (noise floor: 0.0002) | 0.0051 | 0.0058 |
| KLD p90 (noise floor: 0.0050) | 0.0917 | 0.0893 |

`2.81bpw` and `2.9bpw` were measured. turboderp's three packs were not measured on this test set; his
[model card](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) has his KLD and perplexity charts for
them, and his AIME-25 and GPQA-Diamond scores.

The PPL and KLD figures come from one run of exllamav3's qbench. It compared the release's own FP8/FP4 weights, run in
exllamav3, with each quantized pack, on the same tokens and against the same reference. The KLD was measured with
exllamav3's engine, so mlx-serve's runtime, for example FLASH_ATTN's prefill, was not part of it.

- `2.81bpw` was measured on its raw exllamav3 form: the same quantized weights before they were re-laid for
  streaming, no longer published.
- `2.9bpw` was measured on its EXL3 source,
  [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
  at revision `45ee5d0f`. The branch holds that repo's routed experts byte for byte, while its other weights come from
  the release as exact MLX repacks. That repo has no Engram tables, so its run used the same release FP8 tables as the
  2.81 bpw pack.

The test set is a held-out set of 20 conversations, built from turboderp's general prompts with 30% of them using
tools, at medium reasoning effort. It holds 7,109 prompt tokens and 45,434 completion tokens, and FP8 hosts of the
original model generated the completions. The calibration data was a separate trace of 217 rows.

KLD is KL(release ‖ pack) in nats over the completion tokens, with the full 129,280-token vocabulary. The noise floor is
the release itself with BF16-scale rounding noise added after every layer, and every pack's KLD includes it.

### KLD by the release's confidence in its top token

![KLD by the release's top-token confidence, 2.81bpw and EAQ v1 2.9](assets/kld_by_confidence.png)

*The chart shows the mean and the median KLD of each pack in each bucket of the release's top-token probability, with
each bucket's share of the tokens, on a log scale.*

| release top-token probability | share of tokens | `2.81bpw`: mean / median | EAQ v1 2.9 bpw: mean / median |
|---|---|---|---|
| < 0.25 | 1.4% | 0.0773 / 0.0512 | 0.0820 / 0.0534 |
| 0.25 to 0.50 | 13.8% | 0.0843 / 0.0411 | 0.0870 / 0.0432 |
| 0.50 to 0.75 | 19.1% | 0.0823 / 0.0294 | 0.0814 / 0.0290 |
| 0.75 to 0.95 | 17.6% | 0.0517 / 0.0150 | 0.0514 / 0.0153 |
| ≥ 0.95 | 48.0% | 0.0042 / 0.00001 | 0.0048 / 0.00001 |

The paired difference, the 2.81 bpw pack minus EAQ v1 2.9 bpw, is -0.00048 KLD (-1.2%), with a 95% CI of -0.0025 to
+0.0016 from a bootstrap over the 20 conversations. The 2.81 bpw pack is lower on 10 of the 20 conversations. The
result is a tie in quality, at about 2.7% fewer bytes of main model weights for the 2.81 bpw pack.

### Mia-AiLab's own published figures

Mia-AiLab's model card reports the 2.9 bpw pack's fidelity on its own test sets, measured in its own exllamav3 port: KL
0.0137 nats (95% interval 0.0115 to 0.0165) and a top-1 agreement of 96.6% on 100 short conversations of third-party
text (89,880 scored tokens), and KL 0.0051 to 0.0060 on 8 synthetic agentic sessions of 8-31k tokens. Their test text
differs from ours: on the set above the same EAQ pack measures 0.0401, so that set is about three times harder for both
packs. Compare packs within one table, not across them.

## Sweep results

`2.81bpw` and turboderp's three packs ran a calibrated sweep at nine prompt sizes, from 1K to 256K tokens, on an Apple
M5 Max with 128 GB. Speed, latency and hit rate figures are medians of 4 fresh servers per size, and the memory figures
are the highest of the 5 servers that ran at each size. `2.9bpw` was not part of the sweep: its only served run is one
smoke test on mlx-stream, which its
[branch card](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/blob/2.9bpw/README.md)
records. [How the sweep was measured](#how-the-sweep-was-measured) gives the method.

### Speed

![Speed by prompt size for each pack: cold prefill, greedy decode and sampled card decode, 1K to 256K](assets/speed_curve.png)

#### Cold prefill (tok/s)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 176.3 | 528.6 | 293.6 | 200.8 |
| 2K | 312.6 | 827.0 | 510.0 | 358.8 |
| 4K | 581.1 | 1,230.1 | 912.3 | 669.5 |
| 8K | 1,096.6 | 1,686.9 | 1,548.8 | 1,228.8 |
| 16K | 1,723.1 | 1,918.4 | 1,843.0 | 1,713.4 |
| 32K | 1,694.2 | 1,872.6 | 1,789.8 | 1,678.5 |
| 64K | 1,623.8 | 1,800.6 | 1,720.7 | 1,616.1 |
| 128K | 1,538.8 | 1,688.3 | 1,619.6 | 1,523.0 |
| 256K | 1,390.7 | 1,516.2 | 1,454.3 | 1,384.5 |

#### Greedy decode (tok/s)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 58.1 | 72.7 | 68.6 | 62.8 |
| 2K | 58.9 | 72.6 | 72.2 | 62.4 |
| 4K | 57.7 | 70.4 | 68.4 | 60.5 |
| 8K | 61.3 | 72.2 | 69.3 | 60.0 |
| 16K | 59.1 | 71.9 | 68.3 | 58.1 |
| 32K | 59.9 | 69.1 | 69.5 | 57.5 |
| 64K | 58.9 | 68.0 | 67.5 | 60.3 |
| 128K | 59.2 | 69.6 | 65.5 | 59.9 |
| 256K | 57.1 | 67.8 | 65.4 | 57.6 |

#### Card decode, sampled (tok/s)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 49.3 | 69.1 | 65.8 | 50.7 |
| 2K | 48.0 | 71.1 | 65.7 | 51.9 |
| 4K | 47.0 | 71.6 | 64.9 | 50.9 |
| 8K | 48.0 | 70.7 | 61.1 | 50.6 |
| 16K | 49.9 | 65.7 | 62.6 | 52.0 |
| 32K | 48.4 | 67.9 | 60.7 | 49.7 |
| 64K | 44.7 | 68.4 | 63.7 | 47.9 |
| 128K | 46.9 | 66.1 | 63.1 | 49.3 |
| 256K | 46.8 | 64.0 | 56.4¹ | 45.4 |

¹ At 256K, `2.25bpw`'s sampled text ran 1,276 tokens, against 516–764 tokens at every other point, so its 256K
card decode figure is not comparable with the others.

### Latency

![Latency by prompt size for each pack: time to first token and wall time, for a new prompt and for the same prompt again](assets/latency_curve.png)

These times come from the same requests as the speed tables. Time to first token is mlx-serve's own prefill timer, read
from each request's server log line: mlx-serve defines time to first token as request arrival to first token, and on a
one-user server the two are equal. Wall time is the client's clock, from sending the request to having the full
(non-streamed) response, so it includes the output, whose length varies with the text. A new prompt is the card request
on a fresh server with the pack's files evicted from the page cache, so every new-prompt figure is the cold worst case.
The same prompt again is the greedy request that followed on the same server, with all but the prompt's last token
cached; its outputs ran 512–600 tokens. Each figure is the median of 4 fresh servers.

#### Time to first token, new prompt (s)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 5.81 | 1.94 | 3.49 | 5.10 |
| 2K | 6.55 | 2.48 | 4.01 | 5.71 |
| 4K | 7.05 | 3.33 | 4.49 | 6.12 |
| 8K | 7.47 | 4.86 | 5.29 | 6.67 |
| 16K | 9.51 | 8.54 | 8.89 | 9.56 |
| 32K | 19.34 | 17.50 | 18.31 | 19.52 |
| 64K | 40.36 | 36.40 | 38.09 | 40.55 |
| 128K | 85.18 | 77.63 | 80.93 | 86.06 |
| 256K | 188.50 | 172.90 | 180.25 | 189.34 |

#### Wall time, new prompt (s), with output tokens

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 18.0 (588) | 10.4 (574) | 12.9 (610) | 16.9 (588) |
| 2K | 18.2 (547) | 10.3 (552) | 12.2 (531) | 16.3 (539) |
| 4K | 18.5 (526) | 13.1 (696) | 15.6 (714) | 16.9 (534) |
| 8K | 22.3 (687) | 12.7 (549) | 14.3 (539) | 17.3 (529) |
| 16K | 21.9 (600) | 18.0 (611) | 20.4 (718) | 24.6 (764) |
| 32K | 30.8 (540) | 25.4 (530) | 27.5 (545) | 31.1 (561) |
| 64K | 52.3 (516) | 46.4 (670) | 47.1 (561) | 53.5 (606) |
| 128K | 98.0 (584) | 85.9 (542) | 90.4 (583) | 97.4 (543) |
| 256K | 201.0 (562) | 182.6 (605) | 203.1¹ (1,276) | 201.9 (556) |

¹ `2.25bpw`'s 256K text ran 1,276 tokens, against 516–764 at every other point.

#### Same prompt again: time to first token (ms)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 76.8 | 41.3 | 47.9 | 68.0 |
| 2K | 73.2 | 41.2 | 45.0 | 66.3 |
| 4K | 77.0 | 41.2 | 44.9 | 68.5 |
| 8K | 71.7 | 41.1 | 50.8 | 65.1 |
| 16K | 69.2 | 41.5 | 52.7 | 66.4 |
| 32K | 72.5 | 42.0 | 48.8 | 63.1 |
| 64K | 80.0 | 42.7 | 47.5 | 69.7 |
| 128K | 81.7 | 43.5 | 54.9 | 70.7 |
| 256K | 88.1 | 51.2 | 55.6 | 74.6 |

#### Same prompt again: wall time (s)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 9.80 | 7.90 | 8.30 | 9.20 |
| 2K | 9.60 | 7.80 | 7.70 | 9.35 |
| 4K | 9.50 | 7.70 | 8.10 | 9.25 |
| 8K | 8.80 | 8.10 | 7.75 | 9.50 |
| 16K | 9.25 | 7.90 | 8.20 | 9.80 |
| 32K | 8.95 | 8.80 | 8.20 | 9.70 |
| 64K | 9.05 | 8.00 | 8.40 | 9.10 |
| 128K | 9.10 | 7.90 | 8.75 | 9.40 |
| 256K | 9.90 | 8.20 | 8.90 | 9.55 |

### Spec-decode hit rate

![Spec-decode hit rate by prompt size for each pack, greedy and sampled card](assets/hitrate_curve.png)

The hit rate is the share of drafted tokens the model accepted: accepted draft tokens ÷ drafted tokens submitted for
verification, per request, from mlx-serve's per-request `[spec-stats]` line. DSpark drafts 5 positions a round, and with
its verify policy off it submits all 5 every round. These are the same requests as the speed and latency tables. The 4
servers at each point gave identical rounds and accepted counts, so every spread is 0. Each pack decodes its own text,
so the packs compare different texts.

#### Greedy (%)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 76.3 | 78.1 | 76.1 | 81.3 |
| 2K | 77.2 | 77.9 | 80.0 | 80.2 |
| 4K | 76.7 | 76.0 | 76.8 | 77.5 |
| 8K | 81.6 | 78.3 | 79.4 | 77.9 |
| 16K | 78.1 | 80.4 | 80.0 | 77.4 |
| 32K | 77.9 | 76.9 | 82.0 | 73.3 |
| 64K | 77.7 | 76.4 | 80.7 | 80.6 |
| 128K | 79.6 | 79.4 | 78.2 | 78.0 |
| 256K | 77.3 | 77.3 | 79.1 | 78.0 |

#### Card, sampled (%)

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 81.6 | 73.5 | 81.0 | 80.7 |
| 2K | 79.6 | 76.2 | 81.3 | 80.0 |
| 4K | 76.7 | 77.5 | 79.3 | 79.1 |
| 8K | 74.2 | 76.5 | 75.6 | 78.1 |
| 16K | 82.7 | 71.3 | 77.0 | 78.7 |
| 32K | 81.1 | 75.7 | 75.8 | 76.9 |
| 64K | 76.6 | 78.7 | 83.1 | 76.3 |
| 128K | 79.2 | 76.8 | 81.6 | 80.7 |
| 256K | 77.9 | 73.2 | 71.9 | 72.1 |

### Memory

![Memory by prompt size for each pack: server peak plus the rest of the machine, and expert-cache rows](assets/memory_curve.png)

The plugin's calibrated memory bill predicts each server's peak memory before it loads, and in the sweep every server's
peak landed within −0.13 to +0.20 GiB of that prediction. A peak here is the server process's peak physical
footprint. The whole-machine figure adds what the rest of the machine held when each window started (11.0 to
12.3 GiB), and it stayed at or under 110.46 GiB, under the bill's 110.5 GiB target. The memory guard's own
whole-machine reading ran up to 2.25 GiB higher at moments in the 1K–64K windows (highest 111.8 GiB, under its 112 GiB
ceiling). Those moments were macOS's Apple Intelligence on-device model service, which took up to 2.4 GiB at a time,
not the server. The expert-cache rows are how many expert rows the bill gave the cache in the prompt phase and in
decode; panel (b) of the chart shows them as a share of each pack's rows at 1K.

#### Server peak (GiB), the server process alone

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 98.14 | 97.92 | 98.26 | 97.84 |
| 2K | 98.20 | 97.98 | 98.31 | 97.90 |
| 4K | 97.80 | 98.05 | 98.38 | 97.96 |
| 8K | 97.95 | 98.18 | 98.16 | 98.07 |
| 16K | 98.18 | 98.15 | 98.40 | 97.96 |
| 32K | 98.53 | 98.54 | 98.53 | 98.27 |
| 64K | 98.46 | 98.65 | 98.69 | 98.58 |
| 128K | 98.36 | 98.25 | 98.42 | 98.13 |
| 256K | 98.25 | 98.16 | 98.07 | 98.30 |

#### Whole machine (GiB): the server peak plus the rest of the machine at window start

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 109.16 | 109.49 | 109.63 | 109.55 |
| 2K | 109.22 | 109.55 | 109.68 | 109.61 |
| 4K | 108.82 | 109.62 | 109.75 | 109.67 |
| 8K | 108.97 | 109.75 | 109.53 | 109.78 |
| 16K | 109.20 | 109.72 | 109.77 | 109.67 |
| 32K | 109.55 | 110.11 | 109.90 | 109.98 |
| 64K | 109.48 | 110.22 | 110.06 | 110.29 |
| 128K | 109.70 | 110.14 | 110.21 | 110.46 |
| 256K | 109.61 | 110.36 | 110.23 | 110.28 |

#### Expert-cache rows, prompt / decode

| Prompt | `2.81bpw` | `1.68bpw` | `2.25bpw` | `2.68bpw` |
|---|---:|---:|---:|---:|
| 1K | 148 / 184 | 270 / 325 | 202 / 248 | 161 / 200 |
| 2K | 147 / 184 | 270 / 325 | 202 / 248 | 161 / 200 |
| 4K | 147 / 183 | 269 / 325 | 202 / 248 | 161 / 200 |
| 8K | 147 / 183 | 269 / 325 | 201 / 247 | 160 / 200 |
| 16K | 146 / 183 | 267 / 324 | 200 / 247 | 160 / 199 |
| 32K | 146 / 182 | 267 / 323 | 200 / 246 | 159 / 199 |
| 64K | 145 / 182 | 266 / 323 | 199 / 246 | 159 / 198 |
| 128K | 145 / 181 | 265 / 320 | 198 / 244 | 158 / 197 |
| 256K | 144 / 179 | 263 / 316 | 197 / 240 | 158 / 195 |

### How the sweep was measured

The sweep ran on an Apple M5 Max with 128 GB under macOS 27.2, with the fans at maximum. The server was mlx-serve
1271abe0 with its host patches (G_KITBM, 46bbf9bd8b4e), and it ran the turboexl3_ds41 plugin with its calibrated memory
bill. `2.81bpw` ran on plugin build ctl60 (dylib 4a99f8fffb2f), and turboderp's three packs ran on tdp5 (dylib
2d9e2cb095e9), which is the same plugin with fractional-layer support. These dylibs are the measured builds, not
what the install steps build. The plugin in CommunityServe/plugins is the merged code at `a939553`, re-ported onto mlx-serve `bc6c5f2c`. In a standard smoke on `2.81bpw` (the coding prompt, greedy, A B B A then B A A B) it matched #35's earlier head `b3961b0` (dylib 241e0e8bb16b) byte for byte: all 8 runs had the same text sha256 and the same spec-stats, and decode, prefill and peak memory were within 0.2%.

At each size, the prompt was the same code task, built up to that length. Each fresh server ran a keyed sampled request,
the card, at temperature 1.0, top-p 0.95 and top-k 20, and then a greedy request, on the same seeds, with up to 16,384
output tokens. Every request stopped on its own, and the DSpark verify policy was off. Cold prefill is the card
request's prompt processing on its fresh server, and the two decode figures are the two requests' decode speeds.

Each speed, latency and hit-rate figure is the median of 4 fresh servers, and each memory figure is the
highest of the 5 servers that ran at its size. Every text was identical across the 4 servers, and each
greedy text was equal to its 1-row reference.

The server ran with `--prefill-chunk 16384`, `--ctx-size` at the prompt size plus 16,384, `MLX_SERVE_CACHE_LIMIT` at
2 GiB, `MLX_SERVE_GPU_CEILING_MB=102400` and a 112 GiB GPU wired limit, with DSpark's verify policy and copy rounds off
(`MLX_SERVE_DSV41_DSPARK_POLICY=0`, `MLX_SERVE_DSV41_DSPARK_COPY=0`) and keyed sampling (`MLX_SERVE_KEYED_SAMPLING=1`,
from [ddalcu/mlx-serve#1065](https://github.com/ddalcu/mlx-serve/pull/1065), merged as `e5cdd78` and in 26.11.2). The pack's files were evicted from the page cache before each server and after each request.
The plugin applies its calibrated bill only on an M5 Max with 128 GB, with that prefill chunk, cache limit and wired
limit, and otherwise falls back to its earlier bill.

## Parity of the plugin port

The plugin port at commit `0745d79`, now part of the plugin merged in [CommunityServe/plugins#35](https://github.com/CommunityServe/plugins/pull/35), serves
the same texts as the measured build, and it is within 0.5% of its speed. These are 16K windows on `2.81bpw`, in tok/s:

| | a, the measured build (ctl60) | b, the port (`0745d79`) | b − a |
|---|---|---|---:|
| prefill, cold 16K | 1739.6 / 1736.2 | 1743.9 / 1742.7 | +0.31% |
| prefill, new text | 1811.2 / 1822.1 | 1827.2 / 1822.6 | +0.45% |
| decode, greedy | 60.2 / 59.9 / 59.4 / 60.0 | 60.0 / 59.5 / 59.9 / 60.1 | 0.00% |
| decode, sampled (card) | 51.2 / 51.1 / 51.1 / 51.1 | 51.1 / 51.0 / 51.0 / 51.0 | −0.20% |

Both arms ran on one server, mlx-serve 1271abe0 with its host patches, on the same Mac under macOS 27.2, with the
calibrated bill on. Arm a is plugin build ctl60, the build the sweep measured, and arm b is the port at `0745d79`, built
against mlx-serve db128461's kit. The prefill cells are the 16K-token code prompt, cold on a fresh server, and then a
14,917-token slice of new text on the same server. The decode cells are the 16K card, keyed and sampled, and then
greedy, with the DSpark verify policy off. The prefill window ran two unmeasured servers first, one per arm, and then
A B B A; the decode window ran one unmeasured server first, and then A B B A B A A B. b − a is the b mean against the a
mean. The texts were identical in every arm of both windows, with the same bank rows (149 prompt, 186 decode).

The port also ran on stock mlx-serve db128461, one server in each window, without the three host additions of
[ddalcu/mlx-serve#1065](https://github.com/ddalcu/mlx-serve/pull/1065) (merged as `e5cdd78`; in 26.11.2, not in 26.11.1). Without them, BANK_POOL is off, MLX uses
its own command-buffer split and the host uses its default sampler. Against the a arms, the stock server measured:

| | the port on stock db128461, tok/s | against the a arms |
|---|---:|---:|
| prefill, cold 16K | 1693.8 | −2.54% |
| prefill, new text | 1766.5 | −2.76% |
| decode, greedy | 59.9 | +0.04% |
| decode, sampled | 49.1 | −3.96% |

The default sampler drew different tokens after the prefill prompts and in the sampled decode, so the sampled decode is
not a like-for-like speed. Greedy decode produced the same text.

## How the packs were made

The repacks change only the layout: each branch holds its source's quantized tensors byte for byte, and nothing is
quantized again. Each branch's card lists every file and how it was built and checked, and its receipts record the
checks.

- **`2.81bpw`** is repacked from our raw exllamav3 quant, which is no longer published.
  The routed experts are in `experts.bin`, one 4 KiB-aligned record per layer and expert (`a4k`), and the Engram tables
  are the release's, re-laid as one-read records in `engram/`.
  [Branch card](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/blob/2.81bpw/README.md).
- **`1.68bpw`, `2.25bpw` and `2.68bpw`** are repacked the same way from turboderp's branches, checked against the sha256
  in his repo's tree at the revisions above. They carry the same `engram/` files as `2.81bpw`. Branch cards:
  [`1.68bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/blob/1.68bpw/README.md),
  [`2.25bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/blob/2.25bpw/README.md),
  [`2.68bpw`](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/blob/2.68bpw/README.md).
- **`2.9bpw`** holds Mia-AiLab's routed experts byte for byte, at K3 with K2 on layers 18-22. Its other weights are
  unchanged in value and come from the release as exact MLX repacks.
  [Branch card](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/blob/2.9bpw/README.md).

## Credits

DeepSeek made the model and released it under the MIT license (`LICENSE`).

turboderp created [exllamav3](https://github.com/turboderp-org/exllamav3), the EXL3 format and the SC quantization
pipeline, which produced the 2.81 bpw experts and trunk. He also made the 1.68, 2.25 and 2.68 bpw quantizations,
[turboderp/DeepSeek-V4.1-Flash-1017-exl3](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3), that the
`1.68bpw`, `2.25bpw` and `2.68bpw` branches hold.

Mia-AiLab (MiaAI Lab) made the EXL3 EAQ v1 2.9 bpw quantization
[DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
(revision `45ee5d0f`), which supplies the routed experts of `2.9bpw` and the 4-bit EXL3 DSpark stage experts in
`2.81bpw`.

The server is [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu. `2.81bpw` and turboderp's packs run on
the turboexl3_ds41 plugin in [CommunityServe/plugins](https://github.com/CommunityServe/plugins), built on
[turboexl3](https://github.com/davidtai/turboexl3), and `2.9bpw` runs on the
[mlx-stream](https://github.com/davidtai/mlx-stream) plugin. The repacks are by David Tai.

## License

DeepSeek released DeepSeek-V4.1-Flash-1017 under the MIT license, and the `LICENSE` file in this repo is that license.
