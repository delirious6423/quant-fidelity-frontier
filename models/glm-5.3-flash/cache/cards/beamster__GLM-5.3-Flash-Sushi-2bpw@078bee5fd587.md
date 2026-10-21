---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- sushi
- exl3
- apple-silicon
- glm5_next
- moe
---

<p align="center"><img src="assets/sushi-horizon.png" alt="sushi" width="100%"></p>

# GLM-5.3-Flash-Sushi-2bpw

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) packed for
[sushi](https://github.com/beamivalice/sushi), a native inference engine for Apple Silicon.
**Target: a 96+ GB Mac.** The pack is 85.2 GB on disk, the smallest GLM-5.3-Flash pack of the family.

| part | how it is stored |
|---|---|
| routed experts (288 per layer, top-8) | EXL3: 2 bits per weight in MoE layers 3 to 44 and the MTP layer |
| dense part (attention, dense FFN, shared experts, embedding) | 5-bit affine, group size 128 |
| lm_head and vision tower | 6-bit affine, group size 128 |
| MTP layer | included, not used by sushi for GLM |
| context | up to 1,048,576 tokens |
| disk | 85.2 GB (79.4 GiB) |

The small router, mixing and KDA projections keep the precision the checkpoint stores.

**sushi only.** This pack needs sushi v1.2.0 or later on Apple Silicon with macOS 26.2 or later. It does not load in
transformers, vLLM, mlx-lm or exllamav3.

How to run it (install, launch flags, memory and context per Mac, how sushi uses the DFlash2 assistant): [sushi usage](https://github.com/beamivalice/sushi/blob/main/docs/usage.md).

## Included unmodified: the DFlash2 assistant

`GLM-5.3-Flash-DFlash2/` is
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) by
[Inco.ai](https://inco.ai/blog/dflash2/), built on the [DFlash](https://github.com/z-lab/dflash) method, copied
byte for byte from its revision `bf582e4e`. That folder stays under its own license, **CC BY-NC-ND 4.0**
(attribution, non-commercial, no derivatives), not this pack's MIT license. sushi derives a smaller local copy of it
for your Mac (`dflash2/`); that copy is a derivative of a CC BY-NC-ND work: keep it local and never share or upload it.

## Quality

KLD against the BF16 GLM-5.3-Flash checkpoint run through sushi's own forward (BF16 cache, no assistant), with this
pack also at a BF16 cache:
4 prompts x 512 tokens (2,048 positions), two of them code and two prose. Lower is better.

| pack | KLD | top-1 agreement | code KLD | prose KLD |
|---|---|---|---|---|
| **GLM-5.3-Flash-Sushi-2bpw** | **0.103** | **88.3%** | 0.052 | 0.153 |

Four prompts are a screen, not a benchmark.

## Credits and license

- **Z.AI**: the base model, [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), released under the
  **MIT License** (`LICENSE`); this pack is a derivative under the same license.
- **turboderp**: the EXL3 format. The pack contains none of his weights.
- **Inco.ai**: the [DFlash2 assistant](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) for GLM-5.3-Flash
  (CC BY-NC-ND 4.0), and the [DFlash](https://github.com/z-lab/dflash) method behind it. Included unmodified in
  `GLM-5.3-Flash-DFlash2/`, under its own license.
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
