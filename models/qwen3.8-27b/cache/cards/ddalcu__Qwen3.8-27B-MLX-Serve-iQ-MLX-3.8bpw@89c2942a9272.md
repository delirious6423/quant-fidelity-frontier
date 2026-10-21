---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen3_5
- apple-silicon
- imatrix
- mtp
- speculative-decoding
---

# Qwen3.8-27B, iQ-MLX for mlx-serve

Imatrix-calibrated mixed-width MLX conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built for **[mlx-serve](https://mlxserve.com)**, the native Zig MLX server for Apple Silicon.

13.0 GB on disk, 3.78 bits per weight averaged over everything that is quantized. **Text only** — the vision tower is not included.

## Allocation

Per-weight bit widths are chosen by measured error against a byte budget, not picked by hand. Activations were collected on the bf16 weights themselves over 302829 tokens of code, prose, math and real captured agent traffic, and every candidate `(bits, group_size)` was scored by imatrix-weighted reconstruction error; a greedy pass then bought the best error reduction per byte until the budget ran out.

| weight class | params | on disk | widths |
|---|---|---|---|
| MLP gate + up | 11.41B | 5.50 GB | 3-bit/gs-128 x45, 3-bit/gs-64 x7, 4-bit/gs-128 x76 |
| MLP down | 5.70B | 2.75 GB | 3-bit/gs-128 x3, 3-bit/gs-64 x30, 4-bit/gs-128 x31 |
| GDN in_proj_qkv | 2.52B | 1.04 GB | 3-bit/gs-128 x40, 3-bit/gs-64 x8 |
| attention q/k/v/o | 1.68B | 0.94 GB | 4-bit/gs-64 x64 |
| GDN out_proj | 1.51B | 0.70 GB | 3-bit/gs-128 x1, 3-bit/gs-64 x33, 4-bit/gs-128 x14 |
| embed_tokens | 1.27B | 0.68 GB | 4-bit/gs-128 x1 |
| GDN in_proj_z | 1.51B | 0.55 GB | 2-bit/gs-128 x16, 2-bit/gs-64 x1, 3-bit/gs-128 x31 |
| lm_head | 1.27B | 0.52 GB | 3-bit/gs-128 x1 |
| MTP head | 0.37B | 0.21 GB | 4-bit/gs-64 x7 |
| GDN a/b gates | 0.02B | 0.01 GB | 4-bit/gs-64 x96 |
| **total quantized** | **27.27B** | **12.89 GB** | |

## Measured against the 4-bit, 6-bit and 8-bit builds

| build | resident weights | top-1 agreement vs bf16 | mean KL | worst repeated-call run | decode (MTP on) |
|---|---|---|---|---|---|
| [8-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-8bit) | 30.7 GB | 95.5% | 0.014 | 1 | 46.4 tok/s |
| [6-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-6bit) | 22.7 GB | 95.5% | 0.056 | 1 | 57.0 tok/s |
| [4-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) | 17.7 GB | 83.6% | 0.322 | 2 | 69.6 tok/s |
| **iQ-MLX (this one)** | **13.4 GB** | **81.6%** | **0.466** | **1** | **59.0 tok/s** |

**If you have the RAM, download the [6-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-6bit).** It ties the 8-bit on top-1 agreement while being 8 GB smaller and decoding 23% faster, and the 12 points it holds over the 4-bit are far outside the measurement's error bars. The 8-bit only pulls ahead on KL, which is distribution shape rather than argmax, so it shows up at temperature and not at greedy. Take the 4-bit when 22.7 GB will not fit but 17.7 GB will, and take this one when neither will.

Top-1 agreement and KL are against the bf16 source over 32 held-out windows the calibration corpus never saw, 16 sampled positions each. That is 512 sampled positions, so two standard errors is roughly 2 to 3 points and gaps smaller than that are not real: the 4-bit and this build are a tie on top-1, and their 4.3 GB difference is the honest reason to pick between them. The repeated-call column is the longest run of consecutive identical tool calls over three 12-round agent loops against a scripted repository; 1 means it never repeated itself, and this build matches the 6-bit and 8-bit packs there while the 4-bit one does not.

Decode is M4 Max, llmprobe `--bench-only --quick`, all four packs measured in one session with the 4-bit re-measured as a thermal control every four boots and each row normalised to that control. The control moved 5% across the session, so treat differences under about 5% as noise. Without MTP this build is the fastest of the four (33.9 tok/s against 30.0, 21.1 and 16.8) because it reads the fewest bytes. With MTP it drops behind the 4-bit: sub-4-bit weights fall outside `verifyQmmLane`, which is a 4/5/6-bit specialisation, so the verify forward loses its fast lanes and the speculative gain is 1.71x here against 2.27x there.

Attention q/k/v/o, the GatedDeltaNet `in_proj_a`/`in_proj_b` gates and the whole MTP head are pinned at 4-bit/gs-64 — cheap insurance, and the MTP head is the decode lever.

## Serving

```bash
mlx-serve --model ddalcu/Qwen3.8-27B-MLX-Serve-iQ-MLX-3.8bpw --serve --kv-quant 4
```

- **Which Mac.** This build exists for the 24 GB gap. The 4-bit pack needs about
  18 GB of weights resident and does not fit there, and the 8-bit one wants a
  48 GB machine. Resident weights here are **13.4 GB** loaded as shipped, or
  **12.7 GB** with `--no-mtp` (the MTP head is 0.71 GB of that).

  Measured on a real 119,291-token prefill at `--kv-quant 4` with the prefill
  chunk pinned to 512, which is the narrowest rung mlx-serve's sizer can pick and
  therefore the smallest bill this pack can be asked for at that length:

  | flags | peak in use at ~119k |
  |---|---|
  | as shipped | 18.30 GB |
  | `--no-mtp --no-pld --no-drafter` | **17.13 GB** |

  Both answered a fact planted at 97% depth correctly, so that is a working
  context and not just a boot that survived. A 24 GB Mac's Metal working set is
  the term I cannot measure from here: mlx-serve reads it from Metal and falls
  back to 75% of physical, so 18.0 GB, while some 24 GB machines report closer to
  16 GB. At 18.0 GB the `--no-mtp` row fits full 128k with room and the shipped
  row does not. At 16 GB neither reaches 128k and you want roughly 80k. Either
  way `--no-mtp` buys about 1.2 GB, which is the difference between those two
  outcomes, and the numbers above come from a 128 GB Mac, not from a run on a
  24 GB one.

  A 16 GB Mac is NOT covered. The budget that fits one is around 8.6 GB, and at
  that size this model stops following instructions (measured: 46% top-1
  agreement, and two of three agent runs made no tool calls at all).
- **On a 24 GB Mac, consider dropping speculation.** `--no-mtp` costs this build
  less than it costs the others: sub-4-bit weights fall outside `verifyQmmLane`,
  so its speculative gain is 1.71x where the 4-bit pack gets 2.27x. You trade
  roughly 59 tok/s for 34 tok/s and get the 1.2 GB back. Prefill is unaffected.
- **`--kv-quant 4` is half the point.** At fp16 the cache is 64 KB per token
  here; at 4-bit it is 18 KB. mlx-serve sizes its context window accordingly.
- **Thinking** is on by default. `"enable_thinking": false` turns it off; depth is
  `"reasoning_effort": "xhigh" | "medium" | "low"` (Qwen3.8's own vocabulary).
- **Tools** use Qwen3.8's XML call format; mlx-serve parses, repairs and
  schema-coerces it into standard OpenAI `tool_calls`.
- **No images.** This build has no vision tower. Use the
  [4-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit),
  [6-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-6bit) or
  [8-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-8bit) build for that.


## Conversion

- Calibrated affine quantization on every matmul-read weight, widths per the table above.
- Kept bf16: `mtp.fc` and every norm, bias, conv and SSM state. The vision tower is dropped.
- The MTP head ships with the model; mlx-serve finds it in the shards and loads it with no flags.

Same three raw-checkpoint fixes as the [4-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) and [8-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-8bit) builds: the delta-encoded norms get their `+1` folded in, the depthwise conv1d is transposed to MLX's `[C, K, 1]`, and (not needed here, since there is no tower) the Conv3d patch embed is channels-last.

Weights, config, tokenizer and chat template are otherwise verbatim from the base repo.
