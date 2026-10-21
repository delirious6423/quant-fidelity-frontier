---
base_model: zai-org/GLM-5.3-Flash
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - glm
  - exl3
  - tensorfold
  - quantized
  - dgx-spark
---

# GLM-5.3-Flash EXL3 4bpw for TensorFold, by Mia's AI Lab

Mia's AI Lab's own EXL3 quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
built for [TensorFold](https://github.com/ashhart/TensorFold) and served by the
[GLM-5.3-Flash EXL3 TensorFold recipe for DGX Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).

**Lower KL divergence than the widely used TR3-4bpw quant on all six test sets, at the same size, speed and memory,
measured against Z.AI's official FP8 release.** Same format and bit width; only the calibration is new.

| | brandonmusic-TR3-4bpw | This quant |
| --- | --- | --- |
| KL divergence to the FP8 release, as served (wiki / workload) | 0.0990 / 0.3271 | **0.0903 / 0.3150** |
| GSM8K (250, greedy) | 98.0% | 98.8% |
| HumanEval (164, greedy) | 97.6% | 95.7% |
| HumanEval+ / MBPP+ (542, thinking on) | 86.3% | 86.5% |
| Decode, prose, 1 / 4 requests (tok/s, 2x DGX Spark) | 61.5 / 105.5 | 61.8 / 103.2 |
| Prefill, 131k-token prompt (tok/s) | 1,819 | 1,824 |
| Size | 175.7 GB | 175.7 GB |

## Fidelity

![KL divergence reduction by test set](https://huggingface.co/datasets/Mia-AiLab/model-card-assets/resolve/main/glm-5.3-flash-exl3-4bpw-tensorfold/kld_reduction.png)

KL divergence measures how far a quant's next-token probabilities drift from the reference's, Z.AI's FP8 release
(lower is better). This quant lowers it on all six sets: by 11-27% with only the routed experts quantized, and by 2-18% as
TensorFold actually serves it, with its own 4-bit dense weights on top.

As served, part of the remaining error comes from TensorFold's 4-bit dense weights, which no expert quant can remove
(floor: the FP8 release's experts + q4 dense = 0.0676 on wiki, 0.3009 on workload). Of the error the experts themselves add,
this quant removes 28% on wiki and 46% on workload.

![Paired KL divergence change with 95% intervals](https://huggingface.co/datasets/Mia-AiLab/model-card-assets/resolve/main/glm-5.3-flash-exl3-4bpw-tensorfold/kld_paired_ci.png)

Both quants were scored on exactly the same tokens, so the difference can be tested directly. Seven of eight
95% intervals lie entirely below zero; chat is the one set where the gain is within noise.

![Confident mistakes](https://huggingface.co/datasets/Mia-AiLab/model-card-assets/resolve/main/glm-5.3-flash-exl3-4bpw-tensorfold/confident_mistakes.png)

Near ties are coin flips for any quant. The mistakes that matter are where the reference is sure of its next
token, and there this quant picks a different one 16-37% less often (experts only). As served on workload it sits
close to the floor: 1.06% against 1.01% where the reference is over 90% sure.

## Coding

![HumanEval+ and MBPP+, and reply length](https://huggingface.co/datasets/Mia-AiLab/model-card-assets/resolve/main/glm-5.3-flash-exl3-4bpw-tensorfold/coding.png)

On EvalPlus's HumanEval+ and MBPP+, with their stronger tests, thinking on and greedy decoding, both quants solve the same share of 542 problems: 469 against TR3's 468. This quant fixes 14 problems TR3 fails and breaks 13 it solves (paired exact test p = 1.0). It gets there with about 10% shorter replies (median 601 against 664 tokens on MBPP+, 1,020 against 1,139 on HumanEval+), and fewer replies hit the token limit (8 against 11 on MBPP+). Both quants ran on the same TensorFold build with the same settings.

## Served with TensorFold

TensorFold v0.6.0, recipe v1.3.2 defaults (TP=2, 4 streams, 1M-token window, FP8 KV cache, DFlash2 drafts). Both
quants ran on the same build.

| Check | Result |
| --- | --- |
| GSM8K / HumanEval | 98.8% vs TR3's 98.0% (fixes 2, breaks 0) / 95.7% vs TR3's 97.6% (fixes 2, breaks 5) |
| Decode, prose, 1 / 2 / 3 / 4 requests | 61.8 / 79.7 / 92.7 / 103.2 tok/s (TR3: 61.5 / 74.6 / 90.1 / 105.5) |
| Decode, code, 1 / 2 / 3 / 4 requests | 78.6 / 106.1 / 117.8 / 128.3 tok/s (TR3: 79.9 / 107.4 / 118.5 / 124.0) |
| Prefill, 8k-262k tokens | same as TR3 within 1% |
| KV pool | 2,582,528 tokens, same as TR3 |
| 1M-token needle | found |
| Drafted replies equal serial ones | yes, 6/6 |
| Tool calls | OK (array arguments stay JSON arrays) |

Decode figures are means over two interleaved server starts per quant (sparkDash, 512 tokens). The benchmark gaps
are a few problems each and go both ways (GSM8K +2, HumanEval -3; neither is statistically significant): the
KL results above are the measured gain. Both quants' GSM8K and HumanEval runs are on the same build (TR3's replies there are
identical to the recipe's release build); paired exact test: HumanEval p = 0.45, GSM8K p = 0.5.

## What is in the checkpoint

| Part | Format |
| --- | --- |
| Routed experts, layers 3-44 | EXL3, 4 bits a weight (gate, up, down), mcg codebook, output scales |
| MTP layer 45 experts | EXL3, 4 bits a weight |
| Everything else (attention, KDA, shared experts, dense layers, embeddings, head, vision tower) | BF16, unchanged from [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) |
| Chat template | Z.ai's `chat_template.jinja`, the one TensorFold's vision support extends |

Why this layout: TensorFold quantizes the BF16 dense layers itself at load (`DENSE=q4`, `fp8` or `bf16` in the
recipe), and its EXL3 expert kernels read exactly this format, sharded in whole blocks across 2, 3 or 4 Sparks.

Calibration: our own MoE-aware, workload-matched calibration for EXL3. The recipe is not published.

## How it was measured

- Reference: Z.AI's official FP8 release (rev `eb9eb208`, e4m3 with 128x128 block scales, dequantized), full
  vocabulary, temperature 1; reference log-probabilities stored at fp16, KL computed in float32.
- Text: wikitext-2 test (16 x 2,048 tokens) and a workload set of math, code, tool-call and chat conversations in
  GLM's chat template (24 x 4,096 tokens, separate from the calibration data).
- KL divergence = mean over all positions of sum_v p(v) (log p(v) - log q(v)); top-1 = agreement with the
  reference's most likely token. Intervals: 10,000 block-bootstrap resamples of 256-token blocks.
- "As served" replaces the dense layers with TensorFold's own q4 quantization (bit-exact with TensorFold v0.6.0).

## Use

With the [recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) on DGX Sparks.

## License

This quantization is released under the Apache License 2.0 (`LICENSE`). It is derived from GLM-5.3-Flash by
Z.AI, whose MIT license applies to the base model and is kept in `LICENSE-GLM-5.3-Flash`.

## Credits

- Base model: Z.AI, [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).
- Quantization format and converter: [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- Serving engine: [TensorFold](https://github.com/ashhart/TensorFold).
