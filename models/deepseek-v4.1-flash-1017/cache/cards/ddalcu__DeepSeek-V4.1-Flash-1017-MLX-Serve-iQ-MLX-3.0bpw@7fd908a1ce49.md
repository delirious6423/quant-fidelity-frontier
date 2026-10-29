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

# DeepSeek-V4.1-Flash-1017 for mlx-serve, iQ-MLX 3.0 bpw (GPTQ 2/3-bit experts, MXFP8 trunk)

![DeepSeek-V4.1-Flash-1017 iQ-MLX 3.0bpw v1.1: KLD by slice against 1.0, and decode (predictable text and a coding task) and prefill speed by prompt size on an M5 Ultra](https://huggingface.co/ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw/resolve/main/assets/v11-card-chart.png)

mlx-serve pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(`model_type: deepseek_v41`): 40 layers, 384 routed experts (6 active) plus a
shared expert, 1M-token context. Runs on a 256 GB Mac with about 209 GB
resident. Includes the DSpark draft stages (speculative decode, on by default)
and the release's vision tower (image input in mlx-serve 26.10.3 and later).

Every routed expert is quantized with GPTQ on Hessians collected from the
released model's own activations; the attention trunk and shared expert stay
as released (MXFP8). Runs on mlx-serve 26.10.3 or later.

Pick the pack by your Mac's memory:

| pack | resident | download | use it if |
|---|---|---|---|
| **iQ-MLX-3.0bpw (this pack)** | 209 GB | 427 GB | **256 GB Mac** |

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

| version | date | what changed | KLD mean / median | top-1 | perplexity vs released |
|---|---|---|---|---|---|
| **1.1** | 2026-10-29 | Trunk as released (MXFP8) instead of 4-bit; experts re-calibrated on 2.07 M chat-format tokens; act-order GPTQ on the down projection; 3-bit gate/up re-allocated by measured error | **0.0963** / 0.0031 | **92.64%** | +0.6% |
| 1.0 | 2026-10-18 | First release. GPTQ experts on 954 k calibration tokens, 4-bit GPTQ trunk on 131 k tokens | 0.1567 / 0.0067 | 90.59% | -4.2% (see below) |

Both rows are scored the same way, on the evaluation set described below. 1.0's
own card scored it on an earlier set (KLD 0.1044) that is no longer available;
the two sets are different text, so compare rows only within one table.

## Quality

Scored against the released checkpoint's own logits (a bit-exact repack of its
FP4 experts and FP8 trunk) on 46 held-out windows of 2,048 tokens, 11,776
scored positions, through oMLX's DeepSeek-V4.1 port. The windows come from
conversations kept out of calibration: agent sessions with tool calls, code,
tool use, reasoning, chat and multilingual text, rendered with DeepSeek's chat
encoder. Top-1: the pack's greedy token equals the released model's. KLD:
KL(released || pack) over the full 129,280-token vocabulary.

| pack | KLD all | agent | code | tools | reasoning | chat | multilingual | top-1 all |
|---|---|---|---|---|---|---|---|---|
| **1.1 (this pack)** | **0.0963** | **0.169** | **0.046** | **0.045** | **0.019** | **0.085** | **0.117** | **92.64%** |
| 1.0 experts, trunk as released (not published) | 0.1466 | 0.250 | 0.065 | 0.085 | 0.034 | 0.118 | 0.212 | 91.02% |
| 1.0 | 0.1567 | 0.271 | 0.070 | 0.081 | 0.036 | 0.128 | 0.218 | 90.59% |

On the assistant's own tokens (the text the model writes, tool output and
prompts excluded) 1.1's KLD is 0.038 against 1.0's 0.071. Perplexity is not a
quality score here: a pack whose predictions drift from the released model's
can score a lower perplexity than the released model, as 1.0 does; 1.1 stays
within 0.6% of it.

## What changed in 1.1

- **Trunk as released.** The attention projections and the shared expert keep
  the release's MXFP8 instead of 1.0's 4-bit GPTQ. The routed experts are where
  the bytes are; the trunk at 4 bits cost more quality than it saved.
- **Calibration.** 1,012 windows of 2,048 tokens (2.07 M tokens, against 1.0's
  954 k), every window starting at a turn boundary and rendered through
  DeepSeek's encoder, tool calls as DSML blocks: agent trajectories with native
  tool calls (SWE-Gym/OpenHands, SWE-agent), function calling (Hermes), code
  and code reasoning (OpenCodeReasoning), math (OpenR1, verified answers only),
  reasoning (OpenThoughts3), chat (UltraChat), non-English text (Aya),
  WikiText-103 and real source code and docs. Agent and code make up half.
- **Solver.** GPTQ with act-order (columns in descending Hessian order, groups
  fixed so the stored layout is unchanged) on the down projection only; on the
  gate/up projections it measured worse. Damping 0.1, group size 64.
- **Allocation.** Which layers get 3-bit gate/up is solved by a small dynamic
  program over the measured block-output errors at exactly 3.00 bits per expert
  weight: layers 4-9 and 11-14 in 1.1 (1-9 and 14 in 1.0).

Tried for 1.1 and left out, because the end-to-end KLD got worse at the same
size: 4-bit down projections paid for with fewer 3-bit gate/up layers (+7%),
and calibrating the first layers on the quantized model's own drifting
activations (+4%).

## Speed

M5 Ultra (256 GB), one model loaded, measured with mlx-serve main (the build
after 26.10.3: fused DeepSeek-V4.1 decode kernels and a long-context prefill
fix); 1.1 and the unpublished pack above on the same binary, one pass each.

| context | decode, DSpark on | prefill |
|---|---|---|
| ~1 k | 95.8 tok/s | 1,414 tok/s |
| ~8 k | 112.9 tok/s | 1,994 tok/s |
| ~33 k | 108.0 tok/s | 2,276 tok/s |
| ~131 k | 110.2 tok/s | 2,101 tok/s |

The chart at the top is a second llmprobe pass, through the app's own server, from 0.5 k to 66 k tokens, and adds llmprobe's predictable-text decode (128 to 146 tok/s): DSpark's ceiling when every draft lands.

Decode with DSpark on code: 126.5 tok/s. Plain decode on free prose (DSpark
steps aside): 66.7 to 72.4 tok/s. The previous pack on the same build: 89.7 /
105.3 / 89.1 / 98.2 tok/s decode across the same contexts, so 1.1 is at least
as fast (the bytes are the same; DSpark accepts more draft tokens per step when
the pack tracks the released model more closely). mlx-serve 26.10.3 prefills
long prompts slower (848 tok/s at 128 k on this Mac); the fix ships in the next
release.

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
prompt read cold pays for random reads, a warm cache is free. The files are the
same as in 1.0.

## Widths

| tensors | width |
|---|---|
| routed experts gate/up (w1, w3), layers 4-9 and 11-14 | 3-bit, group 64, GPTQ |
| routed experts gate/up, the other 30 layers | 2-bit, group 64, GPTQ |
| routed experts down (w2), all layers | 3-bit, group 64, GPTQ with act-order |
| attention wq_a, wq_b, wkv, wo_b and the shared expert (w1, w2, w3) | MXFP8, as released |
| attention wo_a | bf16 (the release's FP8, widened: exact) |
| routers, indexer, compressor, norms, hyper-connection parameters | bf16 / f32, as released |
| embed_tokens, lm_head | bf16 |
| DSpark draft stages | as released (MXFP4 experts, MXFP8) |
| Engram tables | as released (E4M3 + E8M0), in `engram/` |
| vision tower and aligner | bf16, as released |

Routed experts: 3.00 bits per weight, scales included.

## Serving notes

- **Memory.** About 209 GB resident plus KV cache. mlx-serve sizes the context
  to what fits and refuses a load that would not.
- **DSpark.** On by default; `--no-mtp` or per-request `"enable_mtp": false`
  turns it off. It pays most on code and agent text.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use V4.1's DSML call format; mlx-serve parses and schema-coerces it.
- **LM head.** mlx-serve requantizes the dense head to 8-bit (group 64) at load:
  half the bytes per token, lossy, and not included in the scores above.

## Conversion

GPTQ run layer by layer through oMLX's DeepSeek-V4.1 blocks over the lossless
repack of the released checkpoint, with the calibration streams on SSD, then
packed in oMLX's layout (the pack also loads in oMLX's V4.1 port, which is how
it was scored). The conversion scripts are not in the mlx-serve repo yet.
