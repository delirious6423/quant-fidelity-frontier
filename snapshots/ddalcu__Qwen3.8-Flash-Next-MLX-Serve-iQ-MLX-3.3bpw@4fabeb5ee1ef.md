---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen4_exp
- moe
- sparse-attention
- ngram-embedding
- imatrix
---

# Qwen3.8-Flash-Next for mlx-serve, iQ-MLX 3.3 bpw (64 GB Macs)

mlx-serve pack of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
the Qwen4 preview architecture (`model_type: qwen4_exp`), quantized to 3.3 bits per
weight with imatrix calibration so it fits a 64 GB Mac with room for 64k of context.
52 GB resident. Includes the MTP head and the vision tower (image and video input).

Download [MLXServe.com](https://mlxserve.com)

```bash
sudo sysctl iogpu.wired_limit_mb=58000     # 64 GB Macs, once per boot; see below
mlx-serve --model ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw --serve --ctx-size 65536 --skip-mem-preflight
```
(launching mlx-serve via UI is also fine, make sure to enable --skip-mem-preflight via settings) 


Pick the pack by your Mac's memory:

| pack | resident | download | use it if |
|---|---|---|---|
| **iQ-MLX-3.3bpw (this pack)** | 52 GB | 86 GB | 64 GB Mac |
| [iQ-MLX-4.7bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw) | 75 GB | 107 GB | **128 GB Mac (recommended)** |
| [mixed-4-8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit) | 75 GB | 107 GB | superseded by iQ-MLX-4.7bpw (same size, closer to bf16) |
| [8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-8bit) | 137 GB | 195 GB | more than 128 GB |

Download includes the n-gram table, which is mmapped and not resident (see below).

## Quality

Measured against the bf16 checkpoint on 1200 held-out next-token positions (agent,
code, prose, math; documents the calibration never saw). Top-1 is how often the pack's
greedy token equals bf16's; mass is the bf16 probability of the pack's pick, with
bf16's own pick as the ceiling.

| pack | resident | top-1 vs bf16 | mass (bf16 ceiling 0.762) |
|---|---|---|---|
| mixed-4-8bit | 73 GB | 89.1% | 0.738 |
| **iQ-MLX 3.3 bpw (this pack)** | **52 GB** | **85.6%** | **0.732** |

Same positions, same server, cold prefills. The 3.3 bpw pack keeps 99% of the
4-8 pack's probability mass at 21 GB less.

Variants tried and rejected: more expert bytes at the cost of a 6-bit spine (worse),
forcing the down projections to 3-bit (a wash), 3.6x more calibration data at 44 GB
(no change in measured error; bytes moved the needle, calibration volume did not).

## Memory on a 64 GB Mac

macOS caps what the GPU may wire at about 75% of RAM (48 GB on a 64 GB Mac), which is
not enough. Raise it once per boot:

```bash
sudo sysctl iogpu.wired_limit_mb=58000
```

With that, 52 GB of weights plus a 64k KV cache (1.6 GB) plus the prefill working set
fit. Measured on this pack: 52.2 GB active after load, 55.6 GB peak during a 36k-token
prefill. Keep `--ctx-size` at 65536; `--kv-quant 8` halves the cache if you need more
headroom. The 32 GB n-gram table is not in that number (next section).

## What is different about this model

This is not a Qwen3.5-style pack. Three things around the usual GDN + MoE trunk:

- **Gated residual streams.** The residual is 4 streams wide (4 x 2560). Every
  block reads a sigmoid-mixed average of the normalized streams and writes back
  through per-stream scalar gates. The final mixer replaces the usual final norm.
- **N-gram embedding (51B parameters).** A second embedding table indexed by
  hashed bigrams and trigrams of the token ids: 16 heads, each a prime-sized
  bucket space of ~20M rows, 160 dims per row, injected once before layer 1.
  It is a lookup, no compute, which is why Qwen quotes the model as 125B: the
  full checkpoint is 125B trunk + 51B n-gram + 4B MTP = 180B (360 GB bf16).
- **Qwen Sparse Attention.** Past 2048 tokens each attention layer only reads
  the 512 most relevant 4-token blocks per query (picked by a small indexer),
  plus the query's own partial block. Attention cost stays flat with context.
  Native 262k context.

## How this pack stores the n-gram table

The 51B table is NOT in the safetensors shards. It is one merged 4-bit table
in `ngram_table.bin` (32.0 GB, safetensors format, `.bin` so nothing
mlx-loads it). mlx-serve mmaps the file and, per token, dequantizes the 16 rows
it needs on the CPU (16 x 80 bytes) and uploads only the resulting 2560-vector.
The table never becomes resident: its cost is page cache, which the OS evicts
as needed. On a 64 GB Mac that page cache competes with everything else, so the
first long prompt after boot pays random SSD reads (a few seconds per 8k tokens on
a cold cache); warm is free.

## Widths

| tensors | width |
|---|---|
| routed experts (512 x 48 layers, the 121B) | per-layer, imatrix-measured allocation on layers 0-45: 3-bit group 128 on 67 gate-up/down groups, 2-bit group 128 on 23, 2-bit group 64 on 2; 4-bit group 64 on layers 46-47 and the MTP head. 3.14 bpw over the experts |
| attention, GDN, hyper-connections, indexer, shared experts | 8-bit, group 64 |
| lm_head | 8-bit, group 64 |
| embed_tokens | 4-bit, group 64 |
| n-gram table | 4-bit, group 32 (row width 160) |
| routers, inject gates, norms, convs, SSM state | bf16 |
| MTP head | experts 4-bit group 64, projections 8-bit group 64 |

Every quantized weight with an imatrix entry uses an activation-weighted (scale, bias)
search instead of min-max, byte-compatible with `mx.quantize`'s affine layout, so the
engine reads it like any other affine pack. Every `(1 + w)` RMSNorm has the `+1`
folded into the stored weight; depthwise convs are transposed to MLX's `[C, K, 1]`;
`experts.gate_up_proj` is split into `switch_mlp.gate_proj` / `up_proj`. The vision
tower ships dense bf16 in `model-vision.safetensors` (~0.9 GB).

## Serving notes

- **Speed.** M4 Max, llmprobe `--bench-only --rungs 2k`: 52.6 tok/s decode, 762 tok/s
  prefill, 318 ms TTFT; the mixed-4-8bit pack measured 55.5 / 754 / 278 ms in the same
  run. 3-bit experts ride MLX's stock gather kernel, 2-bit ones mlx-serve's fused ones.
- **MTP.** The checkpoint's own 1-layer speculative head is in the pack (`--mtp` or
  per-request `"enable_mtp": true`). Default-off; not re-measured on this pack.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use Qwen3.8's XML call format; mlx-serve parses and schema-coerces it.
- **Images and video** go through the Qwen3-VL-style tower (`model.visual.*`,
  dense bf16). MTP is declined on image turns (serial decode).

## Calibration and conversion

All in the mlx-serve repo, `tests/`:

1. `qwen38_flash_next_imatrix_collect.py` collects per-input-channel activation
   statistics on the exact bf16 checkpoint (482k tokens of agent traffic, code, prose
   and math, rendered with the model's own chat template) by streaming one decoder
   layer at a time through HF transformers, so the 360 GB never has to be resident.
   The same script writes the held-out bf16 logits the quality table is scored on.
2. `qwen38_flash_next_iq_allocate.py` measures the imatrix-weighted error of every
   layer's gate-up and down banks at 2/3/4 bits and spends a 47 GB budget where it
   buys the most.
3. `convert_qwen38_flash_next.py --imatrix --alloc` writes the pack.
4. `qwen38_flash_next_score.py` scores any served pack against the reference.
