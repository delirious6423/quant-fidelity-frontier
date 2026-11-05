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
- expert-pruning
- engram
- dspark
---

# DeepSeek-V4.1-Flash-1017 for mlx-serve, iQ-MLX Pruned192 2.25 bpw (192 of 384 experts, GPTQ 2-bit, MXFP8 trunk)

mlx-serve pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(`model_type: deepseek_v41`) for a **128 GB Mac**: every layer keeps 192 of its
384 routed experts (6 active per token, as released) plus the shared expert,
and the kept experts are quantized to 2 bits with GPTQ. The attention trunk and
shared expert stay as released (MXFP8). 1M-token context, the DSpark draft
stages (speculative decode, on by default) and the release's vision tower are
included. About 92 GB resident. Runs on mlx-serve 26.11.2 or later.

![DeepSeek-V4.1-Flash-1017 iQ-MLX Pruned192 2.25bpw: KLD by slice against the 3.0bpw pack, and decode (predictable text and a coding task) and prefill speed by prompt size on an M5 Ultra](https://huggingface.co/ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-Pruned192-2.25bpw/resolve/main/assets/card-chart.png)

**Pruning is not free.** Half the experts are gone: code, reasoning and tool
calls hold up well, chat loses more, and **non-English text degrades badly**
(see Quality). If your Mac has 256 GB, use the 3.0 bpw pack.

Pick the pack by your Mac's memory:

| pack | resident | download | use it if |
|---|---|---|---|
| [Lossless](https://huggingface.co/ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-Lossless) | 288 GB | 477 GB | 512 GB Mac, or as a reference |
| [iQ-MLX-3.0bpw](https://huggingface.co/ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw) | 209 GB | 427 GB | 256 GB Mac |
| **iQ-MLX-Pruned192-2.25bpw (this pack)** | 92 GB | 296 GB | **128 GB Mac** |

The download includes the Engram tables (203 GB), which are read from the SSD
per token and are not resident (see below). Keep the model on the Mac's
internal SSD.

Download [MLXServe.com](https://mlxserve.com)

```bash
mlx-serve --model ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-Pruned192-2.25bpw --serve
```

## Versions

Each version is a tag on this repo; `main` is the newest. Pin one with
`hf download ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-Pruned192-2.25bpw --revision v1.0`.

| version | date | what changed | KLD mean / median | top-1 |
|---|---|---|---|---|
| **1.0** | 2026-11-05 | First release | **0.3233** / 0.0141 | **85.82%** |

## Quality

Scored the same way as the 3.0 bpw pack: against the released checkpoint's own
logits (a bit-exact repack of its FP4 experts and FP8 trunk) on 46 held-out
windows of 2,048 tokens, 11,776 scored positions, through oMLX's DeepSeek-V4.1
port. The windows come from conversations kept out of calibration: agent
sessions with tool calls, code, tool use, reasoning, chat and multilingual
text, rendered with DeepSeek's chat encoder. Top-1: the pack's greedy token
equals the released model's. KLD: KL(released || pack) over the full
129,280-token vocabulary.

| pack | KLD all | agent | code | tools | reasoning | chat | multilingual | top-1 all |
|---|---|---|---|---|---|---|---|---|
| iQ-MLX-3.0bpw 1.1 | 0.0963 | 0.169 | 0.046 | 0.045 | 0.019 | 0.085 | 0.117 | 92.64% |
| **Pruned192 1.0 (this pack)** | **0.3233** | **0.420** | **0.141** | **0.168** | **0.069** | **0.405** | **0.809** | **85.82%** |

Top-1 by slice for this pack: agent 87.9%, code 88.2%, tools 92.0%, reasoning
91.9%, chat 79.6%, multilingual 65.4%. On the assistant's own tokens (the text
the model writes, tool output and prompts excluded) its KLD is 0.229 (3.0 bpw:
0.038). Scored with the token embedding at the 8 bits this pack stores
(dequantized, so the values are the ones mlx-serve looks up); the lm head is
requantized to 8 bits at load by mlx-serve and is not in the scores, as for the
3.0 bpw pack.

## Which experts are kept

Per layer, the 192 that score highest on an even blend of:

- **routing-weighted output energy** on the calibration corpus (how often the
  router picks an expert, times how much it moves the output), each domain
  normalized and weighted toward agent work, code and tool calls;
- **the released model's own routing over real agent sessions** (assistant
  reasoning, text and tool calls), plus the experts it routes to where a word
  starts to repeat, which keeps the pruned model out of repetition loops.

They carry 82% of the full router's weight on that agent traffic.
`kept-experts.json` lists them by their index in the release; the router keeps
their rows, so routing among the kept experts is the release's own. An earlier
selection by output energy alone scored better on KLD but looped in long agent
sessions; this blend does not.

## Speed

M5 Ultra (256 GB), one model loaded, mlx-serve 26.11.2, llmprobe, one pass; the
3.0 bpw pack on the same build for comparison.

| context | decode, DSpark on | prefill | 3.0 bpw decode | 3.0 bpw prefill |
|---|---|---|---|---|
| ~1 k | 100.7 tok/s | 2,382 tok/s | 103.2 tok/s | 1,663 tok/s |
| ~8 k | 124.6 tok/s | 3,491 tok/s | 113.6 tok/s | 3,104 tok/s |
| ~33 k | 98.6 tok/s | 4,102 tok/s | 103.0 tok/s | 3,619 tok/s |
| ~131 k | 104.9 tok/s | 3,805 tok/s | 83.7 tok/s | 3,332 tok/s |

Decode is about the same as the 3.0 bpw pack: one token reads 6 experts in
either pack, and attention, not the experts, sets the pace. DSpark decode moves
with how many draft tokens a step accepts, which varies from pass to pass.
Prefill is 1.1x to 1.4x faster: a long prompt touches every expert, and this
pack has half as many, at fewer bits.

The chart at the top is a second llmprobe pass on the same build, from 0.5 k
to 66 k tokens, and adds llmprobe's predictable-text decode (141 to 154 tok/s):
DSpark's ceiling when every draft lands. These numbers are from a 256 GB Mac;
on a 128 GB Mac (lower memory bandwidth on most models) expect less.

**On a 128 GB Mac** (M4 Max): the pack loads with 104.8 GB available against
86.7 GB of weights, sizes the full 1M-token context, and answers, writes code and
calls tools with DSpark engaged, with memory pressure normal and no swap. Speed
there, measured with the Engram tables read over a network share from another
Mac (a local SSD will do better, prefill most): about 31 tok/s decode with
DSpark at 1 k to 33 k tokens, about 470 tok/s prefill.

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
`config.json` names them under `omlx_deepseek_v41.engram_tables`. mlx-serve
reads the rows each token needs with `pread` and dequantizes them on the GPU;
the tables never become resident, their cost is page cache. A fast SSD matters:
a long prompt read cold pays for random reads, a warm cache is free. The files
are the same as in the 3.0 bpw and lossless packs.

## Widths

| tensors | width |
|---|---|
| routed experts gate/up/down (w1, w3, w2), 192 per layer | 2-bit, group 128, GPTQ (act-order on down) |
| attention wq_a, wq_b, wkv, wo_b and the shared expert (w1, w2, w3) | MXFP8, as released |
| attention wo_a | bf16 (the release's FP8, widened: exact) |
| routers (the kept experts' rows), indexer, compressor, norms, hyper-connection parameters | bf16 / f32, as released |
| embed_tokens | 8-bit, group 64 |
| lm_head | bf16 (8-bit at load in mlx-serve) |
| DSpark draft stages | experts 2-bit group 64, the rest as released |
| Engram tables | as released (E4M3 + E8M0), in `engram/` |
| vision tower and aligner | bf16, as released |

Kept routed experts: 2.25 bits per weight, scales included (1.125 per weight of
the released 384).

## Serving notes

- **Memory.** About 92 GB resident plus KV cache. mlx-serve sizes the context
  to what fits and refuses a load that would not. Close large apps first.
- **DSpark.** On by default; `--no-mtp` or per-request `"enable_mtp": false`
  turns it off. It pays most on code and agent text.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use V4.1's DSML call format; mlx-serve parses and schema-coerces it.
- **Other engines.** The 2-bit group-128 expert layout needs mlx-serve's fused
  decode kernels (26.11.2 or later); the pack follows oMLX's layout, which is
  how it was scored.

## Conversion

Experts selected as above, then GPTQ (damp 0.1, group 128, act-order on the
down projection) run layer by layer through oMLX's DeepSeek-V4.1 blocks over
the lossless repack of the released checkpoint, on 1,000 windows of 2,048
tokens, with Hessians collected under the pruned router's own routing (each
kept expert sees the tokens it will see in this pack). The conversion scripts
are not in the mlx-serve repo yet.

## License

MIT, as the base model.
