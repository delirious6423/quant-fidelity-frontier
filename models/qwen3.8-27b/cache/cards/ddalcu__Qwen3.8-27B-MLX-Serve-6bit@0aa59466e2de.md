---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-serve
- qwen3_5
- apple-silicon
- vision-language
- mtp
- speculative-decoding
---

# Qwen3.8-27B, MLX 6-bit for mlx-serve

Plain affine 6-bit (group size 64) MLX conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built for **[mlx-serve](https://mlxserve.com)**, the native Zig MLX server for Apple Silicon.

23.2 GB on disk, 6.50 bits per weight averaged over everything that is quantized. Text, vision, tool calling and thinking all work.

## Allocation

Uniform 6-bit, group size 64, on every matmul-read weight — no imatrix calibration. Measured against imatrix-calibrated mixed-width builds at comparable sizes, a uniform width won: quantization error is convex in bits, so lifting every weight off the cliff beats spending extra bits on a subset. Calibration earns its keep below 4 bits, not here. Group size 64 over 128 was measured on the same held-out set — 95.5% top-1 and KL 0.056 against 93.4% and 0.088 — and costs nothing in decode (46.3 against 46.4 tok/s), so the 0.85 GB is worth paying.

| weight class | params | on disk | widths |
|---|---|---|---|
| MLP gate + up | 11.41B | 9.27 GB | 6-bit/gs-64 x128 |
| MLP down | 5.70B | 4.63 GB | 6-bit/gs-64 x64 |
| GDN in_proj_qkv | 2.52B | 2.04 GB | 6-bit/gs-64 x48 |
| attention q/k/v/o | 1.68B | 1.36 GB | 6-bit/gs-64 x64 |
| GDN in_proj_z | 1.51B | 1.23 GB | 6-bit/gs-64 x48 |
| GDN out_proj | 1.51B | 1.23 GB | 6-bit/gs-64 x48 |
| embed_tokens | 1.27B | 1.03 GB | 6-bit/gs-64 x1 |
| lm_head | 1.27B | 1.03 GB | 6-bit/gs-64 x1 |
| MTP head | 0.37B | 0.30 GB | 6-bit/gs-64 x7 |
| GDN a/b gates | 0.02B | 0.02 GB | 6-bit/gs-64 x96 |
| **total quantized** | **27.27B** | **22.15 GB** | |

## Measured against the 4-bit and 8-bit builds

| build | on disk | top-1 agreement vs bf16 | mean KL | worst repeated-call run | decode (MTP on) |
|---|---|---|---|---|---|
| [8-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-8bit) | 31.2 GB | 96.4% | 0.058 | 1 | — |
| **6-bit (this one)** | **23.2 GB** | **95.0%** | **0.102** | **1** | **46.3 tok/s** |
| [4-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) | 18.2 GB | 83.6% | 0.496 | 2 | 66.8 tok/s |
| [iQ-MLX 3.8bpw](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-iQ-MLX-3.8bpw) | 13.0 GB | 79.6% | 0.611 | 1 | — |

Top-1 agreement and mean KL(bf16 || pack) over 64 held-out windows x 32 sampled positions (2048 distributions), standard error 0.4-0.9 points. The repeated-call column is the longest run of consecutive identical tool calls over three 12-round agent loops against a scripted repository. Decode is M4 Max, temperature 0, code prompt, median of 6 across two passes in opposite boot order, from the server's own reported rate. The jump from 4-bit is the largest single step in the table: KL falls 5x, and the code slice goes from 78.9% to 92.6% top-1. It costs about 31% of decode speed, because MLX's 4-bit affine matmul is the width every fast path in the engine is specialised for.

Attention q/k/v/o, the GatedDeltaNet `in_proj_a`/`in_proj_b` gates and the whole MTP head are pinned at 4-bit/gs-64 — cheap insurance, and the MTP head is the decode lever.

## Serving

```bash
mlx-serve --model ddalcu/Qwen3.8-27B-MLX-Serve-6bit --serve --kv-quant 4
```

- **Which Mac.** This build exists for the 24 GB gap: the 4-bit pack needs about
  18 GB of weights resident and does not fit there, and the 8-bit one needs a
  48 GB machine. 13.0 GB of weights plus a ~1.4 GB prefill working set at
  `--prefill-chunk 512` leaves roughly 1.5 GB for the cache inside a 24 GB Mac's
  ~16 GB Metal working set, which is about 85k tokens at `--kv-quant 4`. A 16 GB
  Mac is NOT covered — the budget that fits one is around 8.6 GB, and at that
  size this model stops following instructions (measured: 46% top-1 agreement,
  and two of three agent runs made no tool calls at all). Those figures are
  computed from measurements taken on a 128 GB Mac, not from a run on a 24 GB one.
- **`--kv-quant 4` is half the point.** At fp16 the cache is 64 KB per token
  here; at 4-bit it is 18 KB. mlx-serve sizes its context window accordingly.
- **Thinking** is on by default. `"enable_thinking": false` turns it off; depth is
  `"reasoning_effort": "xhigh" | "medium" | "low"` (Qwen3.8's own vocabulary).
- **Tools** use Qwen3.8's XML call format; mlx-serve parses, repairs and
  schema-coerces it into standard OpenAI `tool_calls`.
- **Images** work on the chat APIs. Send `image_url` content as usual.


## Conversion

- Calibrated affine quantization on every matmul-read weight, widths per the table above.
- Kept bf16: `mtp.fc` and every norm, bias, conv and SSM state, and the whole vision tower.
- The MTP head ships with the model; mlx-serve finds it in the shards and loads it with no flags.

Same three raw-checkpoint fixes as the [4-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) and [8-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-8bit) builds: the delta-encoded norms get their `+1` folded in, the depthwise conv1d is transposed to MLX's `[C, K, 1]` and the Conv3d patch embed is channels-last.

Weights, config, tokenizer and chat template are otherwise verbatim from the base repo.
