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

**Pack revision 2.0.** Lower KLD in every category than revision 1.0, and the routed experts are stored in the
pair-major tile order that sushi streams from the SSD straight into its expert cache.

| part | how it is stored |
|---|---|
| routed experts (40 layers x 384) | STQ (Sushi Trellis Quant) with the PAIR1 2-D codewords, 1.75 average bits, pair-major tile order |
| trunk | STQ with the SPEC1 codebook, 4 bits |
| lm_head | STQ with the SPEC1 codebook table, 6 bits |
| embedding | 4-bit affine |
| Engram tables | as released (FP8 E4M3 with E8M0 scales), read from the SSD and never loaded into GPU memory |
| DSpark (the 3 speculative-decoding stages) | included: stage experts STQ PAIR1 at 2 bits, stage trunk STQ SPEC1 at 4 bits |
| vision tower | included, bf16: image input |
| context | up to 1,048,576 tokens |
| disk | ~309 GiB: 120 GiB of weights + 189 GiB of Engram tables |

**Sushi Engine Only.** This revision needs **sushi v1.5.0 or later** on Apple Silicon with macOS 26.2 or later.
Earlier sushi versions refuse it by name.

- Have revision 1.0? `sushi pull` fetches 2.0: it compares file contents, downloads only what changed and checks
  every file before it replaces the old one.
- Staying on an older sushi? Revision 1.0 stays available under the `v1.0` tag:
  `hf download beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw --revision v1.0`.

How to run it (install, launch flags, memory and context per Mac): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Quality

KLD against the original checkpoint, run by sushi with no lossy step of its own (FP8 trunk and FP4 experts expanded,
full-width KV, DSpark off), on eight categories (code, math, reasoning, design, wikitext, multilingual, story,
science), 2 prompts x 512 tokens each, scored to the first EOS. Lower is better.

| pack | KLD | top-1 agreement | disk |
|---|---|---|---|
| **Sushi-STQ-1.75bpw, revision 2.0** | **0.1576** | 89.13% | 332.1 GB |
| Sushi-STQ-1.75bpw, revision 1.0 | 0.1843 | 88.53% | 332.1 GB |
| oQ2.5e (oMLX, 2.96 bits per weight) | 0.1717 | 89.17% | 342.7 GB |

oQ2.5e's logits were taken through mlx-serve's DeepSeek path, on the same prompts and teacher.

![KLD per prompt category, revision 1.0 to 2.0, against oQ2.5e](assets/kld-v1-to-v2.svg)

| pack | code | math | reasoning | design | wikitext | multilingual | story | science |
|---|---|---|---|---|---|---|---|---|
| **Sushi-STQ-1.75bpw 2.0** | **0.0284** | **0.0480** | 0.0897 | **0.1468** | **0.2545** | **0.2349** | **0.4052** | **0.0786** |
| Sushi-STQ-1.75bpw 1.0 | 0.0357 | 0.0598 | 0.1176 | 0.1593 | 0.2714 | 0.2733 | 0.4878 | 0.0985 |
| oQ2.5e | 0.0298 | 0.0677 | **0.0831** | 0.1474 | 0.3014 | 0.2868 | 0.4152 | 0.0805 |

## Tool calling

Measured on revision 1.0: tool-eval-bench 3.0.0 in hard mode (seed 42, no live tools), served by sushi 1.4.0 with
its defaults (thinking at high, DSpark on) at `--ssd-budget-gb 108`: 92 / 100, 178 of 194 points over 97 scenarios.

![tool-eval-bench 3.0.0 scores by category](assets/tool-eval.png)

## Credits and license

- **DeepSeek-AI**: the base model, [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
  This pack is a derivative under its **MIT License** (`LICENSE`).
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
