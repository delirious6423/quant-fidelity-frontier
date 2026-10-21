---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: mlx-serve
license: mit
license_link: LICENSE
pipeline_tag: text-generation
language:
- en
- zh
tags:
- mlx
- mlx-serve
- deepseek_v41
- moe
- gptq
- engram
- dspark
---

# DeepSeek-V4.1-Flash-1017 for mlx-serve, iQ-MLX 3.0 bpw (GPTQ 2/3-bit experts, 4-bit trunk)

mlx-serve pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(`model_type: deepseek_v41`): 40 layers, 384 routed experts (6 active) plus a
shared expert, 1M-token context. Runs on a 256 GB Mac with about 206 GB
resident. Includes the DSpark draft stages (speculative decode, on by default)
and the release's vision tower, which mlx-serve does not use yet (text only).

Every routed expert and most of the attention trunk is quantized with GPTQ on
Hessians collected from the released model's own activations, so the scales and
roundings follow what the weights actually see. Needs an mlx-serve build with
[PR #911](https://github.com/ddalcu/mlx-serve/pull/911) (the V4.1 decode
kernels and the Engram layout below); releases up to 26.10.2 cannot read that layout.

Pick the pack by your Mac's memory:

| pack | resident | download | use it if |
|---|---|---|---|
| **iQ-MLX-3.0bpw (this pack)** | 206 GB (199 GB with `--no-mtp`) | 425 GB | **256 GB Mac** |

The download includes the Engram tables (203 GB), which are read from the SSD
per token and are not resident (see below). The load needs about 215 GB free:
close large apps first.

Download [MLXServe.com](https://mlxserve.com)

```bash
mlx-serve --model ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw --serve
```

## Versions

Each version is a tag on this repo; `main` is the newest. Pin one with
`hf download ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw --revision v1.0`.

| version | date | what changed | serial decode | DSpark decode code / prose / agent | KLD median / mean | perplexity vs released |
|---|---|---|---|---|---|---|
| **1.0** | 2026-10-18 | First release. GPTQ experts on 954 k calibration tokens, 4-bit GPTQ trunk on 131 k tokens | 69.4 tok/s | 79.6 / 51.9 / 100.4 tok/s | 0.0076 / 0.1044 | +3.6% |

Decode and perplexity are measured as in the sections below. Later versions are
added here as new rows with what they improve.

## Calibration and what it buys

GPTQ replaces an importance matrix here: it keeps a full Hessian per expert and
per projection (the diagonal of that Hessian is the importance matrix), and
compensates each rounding error in the weights that are still unquantized. The
released checkpoint was streamed layer by layer over 954,368 calibration tokens
for the routed experts (466 windows of 2,048 tokens) and 131,072 tokens for the
trunk (64 windows): agent transcripts (captured tool traffic), code (the
mlx-serve sources and SWE-bench Lite patches), prose, math (generated worked
arithmetic) and WikiText-103, rendered with DeepSeek's own chat encoder. Each
token's input is weighted by its router weight; GPTQ uses damping 0.1 and
group size 64.

Quality is scored against the released checkpoint's own logits (a bit-exact
repack of its FP4 experts and FP8 trunk) on 31 held-out windows of 2,048
tokens, 7,936 scored positions, through oMLX's DeepSeek-V4.1 port. Top-1: the
pack's greedy token equals the released model's. KLD: KL(released || pack) over
the full 129,280-token vocabulary. The windows are disjoint from the
calibration windows.

| pack | download | top-1 all / agent / code / math / prose / wiki | KLD median | KLD mean | perplexity (released 3.476) |
|---|---|---|---|---|---|
| [oMLX oQ3e](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ3e-mtp) | 380 GB | 90.7 / 92.8 / 93.2 / 87.5 / 87.8 / 88.9 | 0.0073 | 0.0844 | 3.519 |
| same experts, trunk as released (not published) | 427 GB | 90.3 / 92.3 / 93.1 / 90.5 / 88.1 / 86.1 | 0.0064 | 0.0964 | 3.576 |
| **iQ-MLX-3.0bpw 1.0 (this pack)** | 425 GB | 89.8 / 92.2 / 92.5 / 89.7 / 87.6 / 85.0 | 0.0076 | 0.1044 | 3.603 |

The 4-bit trunk costs 0.008 KLD and 0.8% perplexity against leaving the trunk
as released, and buys 22% decode speed (56.8 to 69.4 tok/s on the M5 Ultra
below). The scores were taken with a dense LM head; mlx-serve serves it as
8-bit at load (see Serving notes), which is not in the numbers above.

Speed on an M5 Ultra (256 GB) with mlx-serve 26.10.2 plus PR #911, three
400-token greedy answers per prompt type (code, prose, agent), the first of each
cold: serial decode (`--no-mtp`) 69.4 tok/s (69.1 to 69.7) on all three. With
DSpark on (the default): code 79.6, prose 51.9, agent 100.4 tok/s. Prefill 497
to 505 tok/s at 8k tokens, 403 to 431 at 1k. Bits per weight, scales included:
3.06 over the 563 B quantized weights (routed experts 3.00).

## What is different about this model

This is not a Llama-style pack. Four things around the usual MoE trunk:

- **Hyper-connections.** The residual is 4 streams wide (4 x 5120), mixed per
  block by a Sinkhorn-normalized split. Attention collapses the streams with the
  previous layer's FFN mix, and the head with the last FFN mix.
- **Compressed sparse attention.** Each layer reads a 128-token window plus the
  512 best compressed blocks picked by a small indexer; K/V and indexer keys
  are shared across layers. Native 1,048,576-token context.
- **Engram n-gram embeddings.** Layers 1 and 14 add hashed n-gram (up to 4)
  lookups into tables of about 384 M rows each (8 heads x 256 dims). A lookup,
  no matmul: it costs bytes read, not compute.
- **DSpark.** Three block-parallel draft stages (blocks of 5 tokens) ship in
  the checkpoint. mlx-serve drafts with them by default; greedy output equals
  serial decode.

## How this pack stores the Engram tables

The two tables are not re-quantized. They are the release's own
`model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`
(E4M3 values with one E8M0 scale per 32), byte for byte, in `engram/` (203 GB);
`config.json` names them under `omlx_deepseek_v41.engram_tables`. mlx-serve reads the
rows each token needs with `pread` and dequantizes them on the GPU; the tables
never become resident, their cost is page cache. A fast SSD matters: a long
prompt read cold pays for random reads, a warm cache is free.

## Widths

| tensors | width |
|---|---|
| routed experts gate/up (w1, w3), layers 1-9 and 14 | 3-bit, group 64, GPTQ |
| routed experts gate/up, the other 30 layers | 2-bit, group 64, GPTQ |
| routed experts down (w2), all layers | 3-bit, group 64, GPTQ |
| attention wq_a, wq_b, wkv, wo_b and the shared expert (w1, w2, w3), layers 3, 8-36, 38, 39 | 4-bit, group 64, GPTQ |
| the same tensors, layers 0-2, 4-7 and 37 | MXFP8, as released |
| attention wo_a | bf16 (the release's FP8, widened: exact) |
| routers, indexer, compressor, norms, hyper-connection parameters | bf16 / f32, as released |
| embed_tokens, lm_head | bf16 |
| DSpark draft stages | as released (MXFP4 experts, MXFP8) |
| Engram tables | as released (E4M3 + E8M0), in `engram/` |
| vision tower and aligner | bf16, as released (not used by mlx-serve) |

The eight layers kept as released (MXFP8) are the ones whose block output moved
most when the trunk was quantized.

## Serving notes

- **Memory.** About 206 GB resident (199 GB with `--no-mtp`) plus KV cache.
  mlx-serve sizes the context to what fits and refuses a load that would not.
- **DSpark.** On by default; `--no-mtp` or per-request `"enable_mtp": false`
  turns it off. It pays on code and agent text and loses to serial decode on
  free prose (51.9 against 69.4 tok/s above), so prose-heavy work runs faster
  with `--no-mtp`.
- **One request at a time.** Requests queue behind the one running; the prefix
  cache resumes the current conversation, so a follow-up turn skips the
  re-prefill.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use V4.1's DSML call format; mlx-serve parses and schema-coerces it.
- **LM head.** mlx-serve requantizes the dense head to 8-bit (group 64) at load:
  half the bytes per token, lossy, and not included in the scores above.
- **Images** are not supported yet: the tower ships in the pack, mlx-serve
  answers an image with a named 400.

## Known limits of 1.0

- Rarely routed experts have thin calibration: across the 15,360 expert
  matrices of the 40 layers, 28% saw fewer than 5,120 routed tokens.
- The trunk was calibrated on 131 k tokens against the experts' 954 k.
- The weakest slice is Wikipedia text (KLD 0.21): recall of rare facts, not
  code or agent work.

## Conversion

GPTQ run layer by layer through oMLX's DeepSeek-V4.1 blocks over the lossless
repack of the released checkpoint, then packed in oMLX's layout (the pack also
loads in oMLX's V4.1 port, which is how it was scored). The conversion scripts
are not in the mlx-serve repo yet.
