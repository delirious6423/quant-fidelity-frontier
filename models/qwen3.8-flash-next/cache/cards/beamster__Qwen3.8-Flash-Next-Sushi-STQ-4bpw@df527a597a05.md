---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- sushi
- stq
- apple-silicon
- qwen4_exp
- moe
---

# Qwen3.8-Flash-Next-Sushi-STQ-4bpw

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) packed for
[sushi](https://github.com/beamivalice/sushi), a native inference engine for Apple Silicon.
**Target: a 96 GB Mac.** It does not fit on a 64 GB Mac; use
[Sushi-STQ-2.75bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw) there.

It replaces [Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) as sushi's 4-bit Qwen pack:
lower KLD in every category, at about the same size.

| part | how it is stored |
|---|---|
| routed experts (48 layers x 512, plus the MTP layer) | STQ (Sushi Trellis Quant) with the SPEC1 codebook table; 4 or 5 bits per expert bank, 4.15 bits per weight on average |
| dense trunk (attention, GDN projections, indexer, shared experts, lm_head) | STQ with the SPEC1 codebook table, 6 bits |
| embedding, hyper-connections, GDN gate projections, the MTP layer's dense part | 8-bit affine |
| n-gram embedding table | bf16 as released, one file `ngram_table.bin`, read from the SSD and never loaded into GPU memory |
| vision tower (Qwen3-VL ViT) | included: image and video input |
| MTP head | included |
| context | up to 1,048,576 tokens (`config.json` sets YaRN x4 over the native 262,144) |
| disk | ~160 GiB: 64.8 GiB of weights + 95.4 GiB n-gram table |

**Sushi Engine Only.** This pack needs **sushi v1.7.0 or later** on Apple Silicon with macOS 26.2 or later. It does
not load in transformers, vLLM, mlx-lm or exllamav3.

How to run it (install, launch flags, memory and context per Mac): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Quality

KLD against the BF16 checkpoint, run by sushi with no lossy step of its own (bf16 KV, f32 GatedDeltaNet state, MTP
off), on eight categories (code, math, reasoning, design, wikitext, multilingual, story, science), 2 prompts x 512
tokens each, scored to the first EOS. Lower is better. Memory is peak GPU memory while scoring.

| pack | KLD | top-1 agreement | memory |
|---|---|---|---|
| **Sushi-STQ-4bpw** | **0.0111** | **96.35%** | 69.4 GB |
| Sushi-4bpw | 0.0138 | 96.27% | |
| Sushi-STQ-2.75bpw, revision 2.0 | 0.0250 | 94.73% | 48.5 GB |

| pack | code | math | reasoning | design | wikitext | multilingual | story | science |
|---|---|---|---|---|---|---|---|---|
| **Sushi-STQ-4bpw** | **0.0032** | **0.0025** | **0.0099** | **0.0081** | **0.0468** | **0.0116** | **0.0151** | **0.0051** |
| Sushi-4bpw | 0.0041 | 0.0028 | 0.0126 | 0.0114 | 0.0526 | 0.0145 | 0.0189 | 0.0073 |
| Sushi-STQ-2.75bpw 2.0 | 0.0050 | 0.0063 | 0.0198 | 0.0156 | 0.0948 | 0.0226 | 0.0438 | 0.0164 |

A category's mean rests on two prompts: read it for where the loss sits, and rank packs on the overall KLD. Both 4-bit
packs scored by sushi 1.7.0.

![KLD per prompt category, Sushi-4bpw (EXL3) to Sushi-STQ-4bpw, against oQ4e](assets/kld-exl3-to-stq-4bpw.svg)

## Credits and license

- **Qwen team**: the base model, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This
  pack is a derivative under the **Qwen Community License 1.0** (`LICENSE`), including its conditions on large
  commercial services.
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
