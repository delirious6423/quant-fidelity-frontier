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

| part | how it is stored |
|---|---|
| routed experts (48 layers x 512, plus the MTP layer) | STQ (Sushi Trellis Quant) with the SPEC1 codebook table; mixed rates per expert bank from 2 to 4 bits, 2.75 bits per weight on average |
| dense part (attention, GDN, shared experts, hyper-connections, indexer, embedding, lm_head) | 8-bit affine |
| n-gram embedding table | 4-bit affine, group size 32, one file `ngram_table.bin`, read from the SSD and never loaded into GPU memory |
| vision tower (Qwen3-VL ViT) | included: image and video input |
| MTP head | included |
| context | up to 1,048,576 tokens (`config.json` sets YaRN x4 over the native 262,144) |
| disk | ~76 GiB: 46.1 GiB of weights + 29.8 GiB n-gram table |

**sushi only.** This pack needs sushi v1.3.0 or later on Apple Silicon with macOS 26.2 or later. It does not load in
transformers, vLLM, mlx-lm or exllamav3.

How to run it (install, launch flags, memory and context per Mac): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Quality

KLD against the BF16 checkpoint, every pack run by sushi with bf16 KV and `--qwen-gdn 32`. Lower is better.

| reading | KLD | top-1 agreement |
|---|---|---|
| 16 prompts x 512 tokens, scored to the first EOS | 0.0692 | 92.76% |
| eight categories (code, math, reasoning, design, wikitext, multilingual, story, stem), 16 x 512 | 0.0295 | 94.73% |

The eight-category reading spends the same 16 x 512 budget on two prompts per category. On that fixture, at the same
settings, the other Sushi packs and two outside packs read (memory is peak GPU memory while scoring):

| pack | memory | eight-category KLD |
|---|---|---|
| Sushi-2bpw | 37.4 GB | 0.0673 |
| Sushi-2.6bpw | 47.0 GB | 0.0342 |
| **Sushi-STQ-2.75bpw** | **49.6 GB** | **0.0295** |
| Sushi-4bpw | 68.2 GB | 0.0135 |
| oQ4e (oMLX), repacked for sushi | 74.1 GB | 0.0279 |
| mlx-serve mixed 4-8 bit | 75.1 GB | 0.0120 |

KLD by category against Sushi-2bpw and the two outside packs:

![KLD per prompt category for Sushi-2bpw, Sushi-STQ-2.75bpw, oQ4e and mlx-serve mixed 4-8 bit](assets/kld-by-category.png)

Overall KLD of each pack under both readings (compare packs along one line, not across lines):

![Overall KLD per pack under the eight-category average and the 16 x 512 reading](assets/kld-overall.png)

The two readings use different prompts and are not comparable with each other, nor with the kv8 table on the other
packs' cards.

## Credits and license

- **Qwen team**: the base model, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This
  pack is a derivative under the **Qwen Community License 1.0** (`LICENSE`), including its conditions on large
  commercial services.
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
