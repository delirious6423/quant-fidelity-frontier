---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- sushi
- stq
- apple-silicon
- deepseek_v41
- moe
---

# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) packed for
[sushi](https://github.com/beamivalice/sushi), a native inference engine for Apple Silicon.

| part | how it is stored |
|---|---|
| routed experts (40 layers x 384) | STQ (Sushi Trellis Quant) with the PAIR1 2-D codewords, 1.75 average bits | STQ with the SPEC1 codebook, 4 bits |
| lm_head | STQ with the SPEC1 codebook table, 6 bits |
| embedding | 4-bit affine |
| Engram tables | as released (FP8 E4M3 with E8M0 scales), read from the SSD and never loaded into GPU memory |
| DSpark (the 3 speculative-decoding stages) | included: stage experts STQ PAIR1 at 2 bits, stage trunk STQ SPEC1 at 4 bits |
| vision tower | included, bf16: image input |
| context | up to 1,048,576 tokens |
| disk | ~309 GiB: 120 GiB of weights + 189 GiB of Engram tables |

**Sushi Engine Only.** This pack needs sushi v1.4.0 or later on Apple Silicon with macOS 26.2 or later.

How to run it (install, launch flags, memory and context per Mac): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Quality

KLD against the original checkpoint, run by sushi with no lossy step of its own (FP8 trunk and FP4 experts expanded,
full-width KV, DSpark off). Lower is better.

| reading | KLD | top-1 agreement |
|---|---|---|
| 16 prompts x 512 tokens, scored to the first EOS | 0.2197 | 85.64% |
| eight categories (code, math, reasoning, design, wikitext, multilingual, story, science), 16 x 512 | 0.1843 | 88.53% |

On the same fixtures and teacher, oQ2.5e (oMLX, 2.96 bits per weight, 342.7 GB on disk, its logits taken through mlx-serve's DeepSeek path)
reads 0.2575 (85.33%) on the 16 x 512 reading and 0.1717 (89.17%) on the eight categories. This pack is 332.1 GB on disk.

KLD by category, to the first EOS:

![KLD per prompt category for Sushi-STQ-1.75bpw and oQ2.5e](assets/kld-by-category.png)

| pack | code | math | reasoning | design | wikitext | multilingual | story | science |
|---|---|---|---|---|---|---|---|---|
| **Sushi-STQ-1.75bpw** | 0.0357 | **0.0598** | 0.1176 | 0.1593 | **0.2714** | **0.2733** | 0.4878 | 0.0985 |
| oQ2.5e | **0.0298** | 0.0677 | **0.0831** | **0.1474** | 0.3014 | 0.2868 | **0.4152** | **0.0805** |

The two readings use different prompts and are not comparable with each other.

## Tool calling

tool-eval-bench 3.0.0 in hard mode (seed 42, no live tools), served by sushi
1.4.0 with its defaults (thinking at high, DSpark on) at `--ssd-budget-gb 108`: 92 / 100, 178 of 194 points over 97
scenarios.

![tool-eval-bench 3.0.0 scores by category](assets/tool-eval.png)

## Credits and license

- **DeepSeek-AI**: the base model, [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
  This pack is a derivative under its **MIT License** (`LICENSE`).
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
