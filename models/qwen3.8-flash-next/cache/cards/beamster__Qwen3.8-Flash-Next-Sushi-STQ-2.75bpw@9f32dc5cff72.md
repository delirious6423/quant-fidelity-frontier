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

# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) packed for
[sushi](https://github.com/beamivalice/sushi), a native inference engine for Apple Silicon.

**Pack revision 2.0.** Lower KLD in every category than revision 1.0: the dense trunk moves from 8-bit affine to
6-bit STQ, and the pack is about 1 GB smaller.

| part | how it is stored |
|---|---|
| routed experts (48 layers x 512, plus the MTP layer) | STQ (Sushi Trellis Quant) with the SPEC1 codebook table; mixed rates per expert bank from 2 to 4 bits, 2.75 bits per weight on average |
| dense trunk (attention, GDN projections, indexer, shared experts, lm_head) | STQ with the SPEC1 codebook table, 6 bits |
| embedding, hyper-connections, GDN gate projections, the MTP layer's dense part | 8-bit affine |
| n-gram embedding table | 4-bit affine, group size 32, one file `ngram_table.bin`, read from the SSD and never loaded into GPU memory |
| vision tower (Qwen3-VL ViT) | included: image and video input |
| MTP head | included |
| context | up to 1,048,576 tokens (`config.json` sets YaRN x4 over the native 262,144) |
| disk | ~75 GiB: 45.2 GiB of weights + 29.8 GiB n-gram table |

**Sushi Engine Only.** This revision needs **sushi v1.5.1 or later** on Apple Silicon with macOS 26.2 or later. It
does not load in transformers, vLLM, mlx-lm or exllamav3.

- Have revision 1.0? `sushi pull` fetches 2.0: it compares file contents, downloads only what changed (about 45 GB;
  the n-gram table stays) and checks every file before it replaces the old one.
- Staying on an older sushi? Revision 1.0 stays available under the `v1.0` tag:
  `hf download beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw --revision v1.0`.

How to run it (install, launch flags, memory and context per Mac): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Quality

KLD against the BF16 checkpoint, run by sushi with no lossy step of its own (bf16 KV, f32 GatedDeltaNet state, MTP
off), on eight categories (code, math, reasoning, design, wikitext, multilingual, story, science), 2 prompts x 512
tokens each, scored to the first EOS. Lower is better. Memory is peak GPU memory while scoring.

| pack | KLD | top-1 agreement | memory |
|---|---|---|---|
| **Sushi-STQ-2.75bpw, revision 2.0** | **0.0250** | 94.73% | 48.5 GB |
| Sushi-STQ-2.75bpw, revision 1.0 | 0.0295 | 94.73% | 49.5 GB |
| oQ4e (oMLX), repacked for sushi | 0.0279 | 95.19% | 74.1 GB |

![KLD per prompt category, revision 1.0 to 2.0, against oQ4e](assets/kld-v1-to-v2.svg)

| pack | code | math | reasoning | design | wikitext | multilingual | story | science |
|---|---|---|---|---|---|---|---|---|
| **Sushi-STQ-2.75bpw 2.0** | **0.0050** | 0.0063 | 0.0198 | **0.0156** | **0.0948** | **0.0226** | 0.0438 | 0.0164 |
| Sushi-STQ-2.75bpw 1.0 | 0.0059 | 0.0086 | 0.0239 | 0.0188 | 0.0987 | 0.0260 | 0.0576 | 0.0196 |
| oQ4e | 0.0067 | **0.0049** | **0.0191** | 0.0162 | 0.1502 | 0.0233 | **0.0332** | **0.0117** |

A category's mean rests on two prompts: read it for where the loss sits, and rank packs on the overall KLD.

## Tool calling

tool-eval-bench 3.0.0 in hard mode (seed 42, no live tools), greedy, thinking at medium, sushi defaults: revision 2.0
scores 94 / 100, 182 of 194 points over 97 scenarios (revision 1.0: 91 / 100, 176 of 194, on sushi 1.4.0).

![tool-eval-bench 3.0.0 scores by category](assets/tool-eval.png)

## Credits and license

- **Qwen team**: the base model, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This
  pack is a derivative under the **Qwen Community License 1.0** (`LICENSE`), including its conditions on large
  commercial services.
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
