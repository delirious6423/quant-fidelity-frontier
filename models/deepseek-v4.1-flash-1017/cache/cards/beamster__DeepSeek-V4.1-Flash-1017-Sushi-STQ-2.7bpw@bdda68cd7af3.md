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

# DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) packed for
[sushi](https://github.com/beamivalice/sushi), a native inference engine for Apple Silicon.

The higher-quality sibling of
[Sushi-STQ-1.75bpw](https://huggingface.co/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw): 60% lower KLD,
sized to sit fully resident on a 256 GB Mac.

| part | how it is stored |
|---|---|
| routed experts (40 layers x 384) | STQ (Sushi Trellis Quant) with the PAIR1 2-D codewords, 2.72 average bits: each layer's experts in 3.5, 3 and 2.5-bit groups, pair-major tile order |
| trunk | STQ with the SPEC1 codebook, 5 bits |
| lm_head | STQ with the SPEC1 codebook table, 6 bits |
| embedding | 4-bit affine |
| Engram tables | as released (FP8 E4M3 with E8M0 scales), read from the SSD and never loaded into GPU memory |
| DSpark (the 3 speculative-decoding stages) | included: stage experts STQ PAIR1 at 2.5 bits, stage trunk STQ SPEC1 at 5 bits |
| vision tower | included, bf16: image input |
| context | up to 1,048,576 tokens |
| disk | ~373 GiB: 184 GiB of weights + 189 GiB of Engram tables |

**Sushi Engine Only.** Runs on **sushi v1.7.3 or later** on Apple Silicon with macOS 26.2 or later; **v1.7.4 decodes it
13-20% faster** on an M5 Ultra.

How to run it (install, launch flags, memory and context per Mac): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Quality

KLD against the original checkpoint, run by sushi with no lossy step of its own (FP8 trunk and FP4 experts expanded,
full-width KV, DSpark off), on eight categories (code, math, reasoning, design, wikitext, multilingual, story,
science), 2 prompts x 512 tokens each, scored to the first EOS. Lower is better.

| pack | KLD | top-1 agreement | disk |
|---|---|---|---|
| **Sushi-STQ-2.7bpw** | **0.0607** | **93.72%** | 400.1 GB |
| Sushi-STQ-1.75bpw, revision 2.0 | 0.1525 | 89.50% | 332.1 GB |

| pack | code | math | reasoning | design | wikitext | multilingual | story | science |
|---|---|---|---|---|---|---|---|---|
| **Sushi-STQ-2.7bpw** | **0.0076** | **0.0178** | **0.0462** | **0.0686** | **0.0777** | **0.0690** | **0.1682** | **0.0278** |
| Sushi-STQ-1.75bpw 2.0 | 0.0240 | 0.0550 | 0.1149 | 0.1655 | 0.1907 | 0.1927 | 0.3984 | 0.0782 |

Both packs were read against the same teacher; the 2.7bpw reading is sushi 1.7.4 on an M5 Ultra (1.7.3 reads 0.0611,
top-1 93.58%).

## Credits and license

- **DeepSeek-AI**: the base model, [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
  This pack is a derivative under its **MIT License** (`LICENSE`).
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
