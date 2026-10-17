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
---

# Qwen3.8-Flash-Next for mlx-serve (4-bit experts, 8-bit rest)

> **Use [Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw) instead.**
> It is this pack re-quantized with an importance matrix collected on the bf16
> model: same layout, same 75 GB, same bytes read per token, closer to bf16
> (KLD median 0.0052 vs 0.0066, top-1 88.6% vs 87.9% against bf16's own
> logits). This pack stays up for existing installs but gets no more updates.
>
> Switching from this pack? `ngram_table.bin` (32 GB) is identical in both.
> Move it into the new pack's folder before downloading and the download
> skips it:
>
> ```bash
> M=~/.mlx-serve/models/ddalcu
> mkdir -p $M/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw
> mv $M/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/ngram_table.bin $M/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw/
> mlx-serve pull ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw
> ```

mlx-serve pack of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
the Qwen4 preview architecture (`model_type: qwen4_exp`). Runs on a 128 GB Mac
with about 75 GB resident. Includes the MTP head and the vision tower (image
and video input).

Pick the pack by your Mac's memory:

| pack | resident | download | use it if |
|---|---|---|---|
| [iQ-MLX-3.3bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw) | 52 GB | 86 GB | 64 GB Mac |
| [iQ-MLX-4.7bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw) | 75 GB | 107 GB | **128 GB Mac (recommended)** |
| **mixed-4-8bit (this pack)** | 75 GB | 107 GB | superseded by iQ-MLX-4.7bpw (same size, closer to bf16) |
| [8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-8bit) | 137 GB | 195 GB | more than 128 GB |

Download includes the n-gram table, which is mmapped and not resident (see below).

Download [MLXServe.com](https://mlxserve.com)

```bash
mlx-serve --model ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit --serve
```

Measured on an M4 Max 128 GB with mlx-serve 26.8.11: ~75 GB resident,
decode ~60 tok/s serial and 78 tok/s with MTP (`--mtp`; +41% on code, a few
percent slower on prose), prefill ~730 tok/s, a needle at 24.8k tokens
recovered with sparse attention engaged. Prefix cache on, images included.

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
as needed. That is the difference between this pack and mlx-lm style packs
that ship the table as 128 quantized tensors and load it onto the GPU
(+32 GB resident, ~107 GB total for a 4-bit pack).

Expected effect: decode speed unchanged (16 tiny reads against a ~20 ms step),
cold-cache prefill of very long prompts may pay up to ~1 s per 8k tokens of
random reads on the SSD, warm cache is free. No user-space cache is needed,
the page cache already is an LRU over exactly this access pattern.

## Widths

| tensors | width |
|---|---|
| routed experts (512 x 48 layers, the 121B) | 4-bit, group 64 |
| attention, GDN, hyper-connections, indexer, shared experts | 8-bit, group 64 |
| lm_head | 8-bit, group 64 |
| embed_tokens | 4-bit, group 64 |
| n-gram table | 4-bit, group 32 (row width 160) |
| routers, inject gates, norms, convs, SSM state | bf16 |
| MTP head | same policy as the trunk |

Every `(1 + w)` RMSNorm has the `+1` folded into the stored weight; depthwise
convs are transposed to MLX's `[C, K, 1]`; `experts.gate_up_proj` is split into
`switch_mlp.gate_proj` / `up_proj`. The vision tower ships dense bf16 in
`model-vision.safetensors` (~0.9 GB).

## Serving notes

- **Memory.** ~75 GB resident plus KV cache. mlx-serve sizes the context to
  what fits; `--kv-quant 8` halves the cache.
- **MTP.** The checkpoint's own 1-layer speculative head is loaded from the
  pack (`--mtp` or per-request `"enable_mtp": true`). Measured on an M4 Max
  vs serial decode: code +41%, prose -4%, an 8.5k-token prompt -4%. It is
  opt-in for now because the win depends on the prompt.
- **v1 limits in mlx-serve.** One request at a time (no batched decode),
  PLD/DFlash speculation off (MTP is the speculative path). Prefix cache is
  on, images included, so follow-up turns skip the re-prefill. Very long
  prompts (past ~64k) want a smaller `--prefill-chunk` because the
  sparse-attention selection is built per chunk.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use Qwen3.8's XML call format; mlx-serve parses and schema-coerces it.
- **Images and video** go through the Qwen3-VL-style tower (`model.visual.*`,
  dense bf16). MTP is declined on image turns (serial decode).

## Conversion

`tests/convert_qwen38_flash_next.py` in the mlx-serve repo. It streams the
360 GB bf16 checkpoint shard by shard from the Hub (download, quantize, delete),
so it converts on a machine with ~150 GB free. The engine was validated against
HF transformers (trunk) and the vLLM/SGLang MTP math on a tiny random model
before the full conversion.
