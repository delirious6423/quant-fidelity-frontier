---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - ornith
  - qwen3_5
  - 35B
  - abliterated
  - uncensored
  - abliterix
  - multimodal
  - vision
  - gguf
  - 8-bit
  - quantized
---

# Ornith-1.5-35B-A3B-uncensored-GGUF-8bit

An **abliterated** (refusal-direction-ablated) **35B** vision-language build of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), produced with
[Abliterix](https://github.com/wuwangzhang1216/abliterix) (winning trial **#17**) and published by
[junafinity](https://huggingface.co/junafinity).

**Vision tower and MTP heads are preserved** — see
[Vision & MTP preservation](#vision--mtp-preservation) for the before/after audit.

This GGUF is the **MTP path** for 35B-A3B (`blk.40` / `nextn` + separate mmproj). The MLX siblings drop MTP.

## Intended use: red teaming and defensive cybersecurity research

These uncensored (abliterated) weights are built as a **research instrument** for red teaming and defensive cybersecurity work. Safety training suppresses the *display* of capability, not capability itself. A refusal tells you the model declined. It does not tell you whether the weights could have complied. That conflation underestimates the true ceiling and hides holes in *your* filters, classifiers, and policy layer.

Use each uncensored checkpoint as the **treatment half of a controlled pair** against its original base model:

- **Capability-ceiling measurement.** Upper-bound what the weights can actually produce in a domain, independent of shipped refusals.
- **Defensive-stack evaluation.** Test input filters, output classifiers, prompt-injection defenses, and moderation APIs when the model itself contributes no refusals. That is how you find gaps in a defensive control plane.
- **Attack-surface isolation.** Automated red-team loops stall on unrelated refusals. A non-refusing target isolates the control under test (injection, tool abuse, data-exfil paths, policy bypass).
- **Detection and classifier work.** Generate labeled completions for training or benchmarking output-moderation and abuse-detection models.
- **Interpretability of residual refusal.** Abliteration is a specified edit on known language-model components. The pair (base vs this) is a clean experimental control.

**Operating rules.** Do not expose these weights as a public endpoint without an independent moderation layer. Abliteration removes a direction, not a policy; some refusals survive (multi-turn re-assertion, system-prompt steering, vision-path refusals). Always report the **delta against the base model**. Re-measure on your own prompts. Whoever deploys it owns the moderation layer the original guardrails were carrying.

## Variants in this family

Hub collection: [https://huggingface.co/collections/junafinity/ornith-15-uncensored-6a896c737cf40ad660af2ebd](https://huggingface.co/collections/junafinity/ornith-15-uncensored-6a896c737cf40ad660af2ebd)

| Model | Base | Format | Precision | Notes |
|---|---|---|---|---|
| [Ornith-1.5-9B-uncensored](https://huggingface.co/junafinity/Ornith-1.5-9B-uncensored) | Ornith-1.5-9B | Safetensors (bf16) | 16-bit | Full-precision abliterated weights |
| [Ornith-1.5-9B-uncensored-MLX-8bit](https://huggingface.co/junafinity/Ornith-1.5-9B-uncensored-MLX-8bit) | Ornith-1.5-9B | MLX | 8-bit | Apple Silicon, `mlx-vlm` |
| [Ornith-1.5-9B-uncensored-GGUF-8bit](https://huggingface.co/junafinity/Ornith-1.5-9B-uncensored-GGUF-8bit) | Ornith-1.5-9B | GGUF | Q8_0 | llama.cpp |
| [Ornith-1.5-35B-A3B-uncensored-MLX-8bit](https://huggingface.co/junafinity/Ornith-1.5-35B-A3B-uncensored-MLX-8bit) | Ornith-1.5-35B-A3B | MLX | 8-bit | Apple Silicon, `mlx-vlm` |
| [Ornith-1.5-35B-A3B-uncensored-MLX-MXFP4](https://huggingface.co/junafinity/Ornith-1.5-35B-A3B-uncensored-MLX-MXFP4) | Ornith-1.5-35B-A3B | MLX | MXFP4 | Apple Silicon, `mlx-vlm` |
| **Ornith-1.5-35B-A3B-uncensored-GGUF-8bit** ← *you are here* | Ornith-1.5-35B-A3B | GGUF | Q8_0 | llama.cpp |

## Vision & MTP preservation

**The vision tower and the multi-token-prediction (MTP) block are not Abliterix steering targets.** The edit touches language-model attention q/k/v/o, `mlp.down_proj`, and fused MoE expert/router parameters. Vision and `mtp.*` tensors are never steered.

| Component | Original checkpoint | This artifact | Status |
|---|---|---|---|
| **Vision tower** | 333 tensors / 446,571,248 params | separate `mmproj-Ornith-1.5-35B-A3B-uncensored-f16.gguf` (334 tensors, 0.90 GB) | ✅ preserved |
| **MTP head** | 785 tensors / 844,640,768 params | ✅ exported as `blk.40` (incl. `nextn.*`) | ✅ re-grafted byte-for-byte from the original |

> **Note on tooling:** `transformers` 5.15.1 has no MTP implementation for `qwen3_5_moe` — a plain load/save round-trip silently drops all 785 MTP tensors. They were re-grafted **byte-for-byte from the original checkpoint** after abliteration, which is correct because the MTP block is never an abliteration target.

> **Format note:** in GGUF the vision tower ships as a separate `mmproj-*.gguf` file (llama.cpp's standard multimodal layout) — download it alongside the model weights to use images. The MTP block is exported into the same GGUF as the final `blk` index (`nextn` tensors), so speculative decoding support is retained.

## Abliteration result

| Metric | Value |
|---|---|
| Refusals on held-out harmful set | **100 → 9** / 100 (**9%**) |
| KL divergence from base | **0.3985** |
| Tool | Abliterix 1.12.2 |
| Optuna trials | 50 (15 warmup), seed 42 |
| Selected trial | **#17** |
| Steering | per-layer attn q/k/v/o + `mlp.down_proj` |
| MoE expert steering | `n_suppress=4`, `router_bias=-2.72`, `expert_ablation_weight=4.31` |

Abliterix searches a two-objective surface — remaining refusals vs. KL from the base — then materializes the winning trial as a weight edit. There is no runtime adapter.

> **These figures were measured on the bf16 parent, not on this quantized checkpoint.** Quantization is applied *after* those measurements. If exact numbers matter for your work, re-run the evaluation against this GGUF.

## Method

1. Residual-stream activations captured on harmful vs. harmless prompt sets.
2. Refusal direction estimated per layer; attention and `mlp.down_proj` steered.
3. Fused-MoE expert suppression + router bias (the path Heretic cannot touch on this architecture).
4. Optuna TPE over 50 trials; trial **#17** selected (9% refusals, KL 0.3985, under the 0.5 damage threshold).

## Usage

```bash
# text
llama-cli -m Ornith-1.5-35B-A3B-uncensored-Q8_0.gguf -p "The capital of Japan is" -n 64

# image + text (download the mmproj file from this repo too)
llama-mtmd-cli -m Ornith-1.5-35B-A3B-uncensored-Q8_0.gguf \
  --mmproj mmproj-Ornith-1.5-35B-A3B-uncensored-f16.gguf \
  --image photo.jpg -p "What is in this image?"
```

Sampling (matches the base Ornith 1.5 card): `--temp 0.6 --top-p 0.95 --top-k 20`.

Two files ship here: the quantized language model and the `mmproj-*` vision
projector. Download **both** for multimodal use; the model file alone is
text-only.

## Responsible use

Primary intended use is **red teaming and defensive cybersecurity research**. See the section of that name above.

This model has had safety guardrails **reduced or removed**. Do not expose it as a public endpoint without an independent moderation layer. You are responsible for compliance with the base model's license and acceptable-use policy, applicable law, and the terms of any platform you deploy on. Removing guardrails does not remove accountability.
