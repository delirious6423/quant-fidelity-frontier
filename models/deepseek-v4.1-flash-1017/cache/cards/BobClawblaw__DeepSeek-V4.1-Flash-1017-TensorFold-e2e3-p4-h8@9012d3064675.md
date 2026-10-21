---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- deepseek
- deepseek-v4
- tensorfold
- metal
- apple-silicon
- m5-ultra
- mixture-of-experts
- quantized
- 2-bit
- 3-bit
- long-context
- speculative-decoding
- vision-language
---

# DeepSeek-V4.1-Flash-1017, TensorFold e2e3-p4-h8

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at `1bae0e10`, converted to
MLX affine codes for TensorFold's native Metal engine on one 256 GB Apple Silicon Mac.

**On one M5 Ultra:**
- **Decode:** 67.5–94.4 tok/s one stream, with DSpark drafts; four streams together 89.8–102.1 tok/s.
- **Cold prompts:** 1,093–1,392 tok/s.
- **Context:** the full 1,048,576 tokens; a 1,007,409-token prompt recalled correctly in 29.7 min.
- **Quality, thinking off:** GSM8K 247/250, MATH-500 at least 465/500, MMLU 307/342, MGSM 217/240, IFEval 129/150 (loose).
- **Quality, thinking on:** GSM8K 243/250, MATH-500 at least 469/500.
- **Also:** images (pasted, or public HTTPS links with the recipe's `vision_urls`), tools and JSON output.

**Serving:** the [deepseek-v41-flash-mlx-m5-ultra](https://github.com/BobClawblaw/deepseek-v41-flash-mlx-m5-ultra) recipe
serves it with the `deepseek-v41-main` branch of [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/deepseek-v41-main)
(not upstream TensorFold yet). The [gist](https://gist.github.com/BobClawblaw/5e120cb3f7a7b3c9d7ca9b40ebea29f0)'s
reference runners also read it. It is not an `mlx-lm` checkpoint.

## Widths

All quantized tensors are MLX affine, group 64.

| Part | Bits |
|---|---|
| Routed experts (`e2e3`): layers 0–9 and 30–39, and the three DSpark draft stages | 2 |
| Routed experts, layers 10–29 | 3 |
| The blocks' and draft stages' dense projections (`p4`): attention `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`; the indexer's `wq_b`; the shared expert; the Engram `wkv`; the draft head's `main_proj` | 4 |
| Head and image tower (`h8`) | 8 |

- **Unquantized:** norms, routers, compressors, the indexers' key and weight projections and the hyper-connection
  tensors keep the release's bf16 or fp32.
- **Layout:** routed experts are stacked per layer as `layers.N.ffn.experts.w{1,2,3}` [384, out, in]. `wo_a` is
  stacked by group, [8, 1024, 4096].
- **Shards:** 201 GB.

## Engram tables

The two Engram tables are the release's FP8 bytes, unchanged, repacked one row a span: `rows` u8 [N, 264], each row
256 e4m3 bytes then 8 e8m0 scales.

- **Files:** each table is split into three consecutive-row parts to stay under the Hub's per-file limit,
  `engram/layer{1,14}-part{0,1,2}.safetensors`, about 34 GB each. The engine reads the parts in order.
- **Hash state:** `engram/hash.safetensors` holds the n-gram hash state.

## Quality

- **KL against the release's own fp4 weights,** teacher-forced on four prompts × 64 tokens of the fp4 model's greedy
  replies: KL 0.138, top-1 agreement 0.909. Measured with the recipe's own tool, not a harness quality suite.
- **Whole-model gate,** with the gist's runners (DeepSeek's reference over these weights):
  - **Eight tokens:** the engine is 8.08% NRMS from the FP64 chain, 1.91× the 4.22% between the FP32 and FP64 chains,
    with top-1 agreement 8/8.
  - **A 268-token greedy reply, replayed through the chains:** the engine's tokens are the FP64 chain's argmax at
    252 of 268 positions; the FP32 chain's own argmax agrees with FP64 at 250.
- **Harness quality suite:** [recipe-harness](https://github.com/BobClawblaw/recipe-harness), greedy, the
  [GLM-5.3 recipe's](https://github.com/BobClawblaw/glm53-flash-exl3-2x-dgx-sparks) suite.
  - **Thinking off:** run `20261018-202801-dsv41-quality`, engine `269f2447`.
  - **Thinking on:** run `20261019-034805-dsv41-thinking`, engine `7837387a` with a 32,768-token thinking reserve.

| Check | Thinking off | Thinking on |
|---|---|---|
| GSM8K (first 250) | 247/250 (98.8%) | 243/250 (97.2%) |
| MATH-500, the suite's fixed grader | at least 465/500 | at least 469/500 |
| MATH-500, as the runs graded it | 457/500 (91.4%) | 451/500 (90.2%) |
| MGSM (8 languages, 30 each) | 217/240 (90.4%) | not run |
| MMLU (342 questions) | 307/342 (89.8%) | not run |
| IFEval (150 prompts) | 129/150 loose, 122/150 strict | not run |
| Tools | 58/60 | not run |
| Structured output | 30/30 | not run |
| GPQA-Diamond | not run | not run |
| The harness's thinking problems | — | 10/10 |

- **MATH-500:** the suite's grader rejected some correct answers. It missed units after a number, `{,}` separators,
  set braces and lists in another order, and had a bug with matrices written with spaces. It was fixed on
  2026-10-19, and both runs were regraded from the answers they kept. Those runs kept only their first 40 misses,
  hence "at least".
- **GSM8K with thinking on:** misses four problems that thinking off gets right, each by reading ambiguous wording
  another way than the answer key.
- **GPQA-Diamond with thinking on:** not run to the end. Under greedy decoding, two of its first twenty questions
  reasoned in a loop and never answered, even with 53,248 tokens. The recipe's loop guard (engine `f8a3ce2f`) now
  closes such a loop: both then answered, wrongly.
- **Tools:** both misses are right calls with one optional argument more than the checker expects.

## Speed

[recipe-harness](https://github.com/BobClawblaw/recipe-harness), target `deepseek-v41-m5`, one M5 Ultra (80-core GPU,
256 GB). DSpark drafts on, thinking off, greedy, 256-token replies to short prompts.

**One stream:** run `20261019-090829-dsv41-speed`, engine `c4c69219`.

| Prose | Code | Structured | List | First token |
|---|---|---|---|---|
| 70.4 tok/s | 90.5 tok/s | 76.8 tok/s | 88.6 tok/s | 0.11–0.14 s |

The earlier engine `6a4d7be7` measured 67.5 / 87.5 / 77.6 / 94.4 (run `20261018-184411-speed`). A lone stream runs
the same arithmetic on both: its logits are byte-identical.

**Several streams:** the same run. Up to four streams' windows go through one forward, and each reply equals the same
request alone. This table counts total tokens over the elapsed time, first tokens included:

| Streams | Prose | Code | Structured | List | First token |
|---|---|---|---|---|---|
| 1 | 64.0 | 86.8 | 71.4 | 84.7 | 0.11–0.14 s |
| 2 | 82.1 | 94.7 | 79.3 | 98.8 | 0.22–0.28 s |
| 4 | 93.5 | 101.4 | 89.8 | 102.1 | 0.45–0.58 s |

- **Four streams give 1.17–1.46x one stream, not 4x:** each row still reads and multiplies its own routed experts.
- **Earlier engines ran streams one after another,** so two streams together decoded at about one stream's speed.
- **A new prompt goes in 1,024 tokens at a time,** with the other streams decoding for half the time between chunks.
  A 60k-token prompt took 54.2 s while another reply decoded, against 43.0 s alone.

**Prompts:** run `20261018-183907-prefill`, engine `6a4d7be7`. One cold prompt per size, timed to the first token
(with thinking on, the first reasoning token).

| Prompt tokens | Thinking off | Thinking on |
|---|---|---|
| ~2k | 1,093 tok/s | 1,339 tok/s |
| ~8k | 1,240 tok/s | 1,392 tok/s |
| ~33k | 1,256 tok/s | 1,334 tok/s |
| ~131k | 1,195 tok/s | 1,210 tok/s |

- **The full window**, through the server, on engine `6a4d7be7`:
  - A 1,007,409-token prompt ran in 1,783.5 s (565 tok/s), with at least 19.5 GiB of memory free.
  - A follow-up turn restored 1,007,407 cached tokens and gave its first token in 7.8 s.
  - It decoded 197 tokens at 46.3 tok/s. That's one sample from a script, not the harness.

## License

MIT, as the release (`LICENSE`, copyright DeepSeek).
