---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: mlx-vlm
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
  - mlx
  - 8-bit
  - apple-silicon
  - quantized
---

# Ornith-1.5-35B-A3B-uncensored-MLX-8bit

An **abliterated** (refusal-direction-ablated) **35B** vision-language build of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), produced with
[Abliterix](https://github.com/wuwangzhang1216/abliterix) (winning trial **#17**) and published by
[junafinity](https://huggingface.co/junafinity).

**Vision tower is preserved** — see
[Vision & MTP preservation](#vision--mtp-preservation). Converted with `mlx-vlm`
(`-q --q-bits 8 --dtype bfloat16`, 8.596 bits per weight).

This is an **`mlx-vlm` vision checkpoint** (tower inside the file). `mlx-vlm` **drops `mtp.*`**. For 35B native MTP use [Ornith-1.5-35B-A3B-uncensored-GGUF-8bit](https://huggingface.co/junafinity/Ornith-1.5-35B-A3B-uncensored-GGUF-8bit).

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
| **Ornith-1.5-35B-A3B-uncensored-MLX-8bit** ← *you are here* | Ornith-1.5-35B-A3B | MLX | 8-bit | Apple Silicon, `mlx-vlm` |
| [Ornith-1.5-35B-A3B-uncensored-MLX-MXFP4](https://huggingface.co/junafinity/Ornith-1.5-35B-A3B-uncensored-MLX-MXFP4) | Ornith-1.5-35B-A3B | MLX | MXFP4 | Apple Silicon, `mlx-vlm` |
| [Ornith-1.5-35B-A3B-uncensored-GGUF-8bit](https://huggingface.co/junafinity/Ornith-1.5-35B-A3B-uncensored-GGUF-8bit) | Ornith-1.5-35B-A3B | GGUF | Q8_0 | llama.cpp |

## Vision & MTP preservation

**The vision tower and the multi-token-prediction (MTP) block are not Abliterix steering targets.** The edit touches language-model attention q/k/v/o, `mlp.down_proj`, and fused MoE expert/router parameters. Vision and `mtp.*` tensors are never steered.

| Component | In this artifact |
|---|---|
| Vision tower | ✅ inside the checkpoint (`mlx-vlm`; preprocessor sidecars included) |
| MTP head | ❌ not carried in this format (see the format note below) |

> **Format note:** the vision tower is carried inside the MLX checkpoint (converted with `mlx-vlm`, which retains it — note that `mlx-lm` would strip it).
>
> ⚠️ **`mlx-vlm` unconditionally drops `mtp.*` tensors during conversion, so this MLX build does not carry the MTP head.** If you need multi-token prediction, use [Ornith-1.5-35B-A3B-uncensored-GGUF-8bit](https://huggingface.co/junafinity/Ornith-1.5-35B-A3B-uncensored-GGUF-8bit), which does.

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

> **These figures were measured on the bf16 parent, not on this quantized checkpoint.** Quantization is applied *after* those measurements. If exact numbers matter for your work, re-run the evaluation against this MLX build.

## Method

1. Residual-stream activations captured on harmful vs. harmless prompt sets.
2. Refusal direction estimated per layer; attention and `mlp.down_proj` steered.
3. Fused-MoE expert suppression + router bias (the path Heretic cannot touch on this architecture).
4. Optuna TPE over 50 trials; trial **#17** selected (9% refusals, KL 0.3985, under the 0.5 damage threshold).

## Usage

Requires **Apple Silicon** (M-series) and [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm):

```bash
pip install mlx-vlm
```

```bash
# text
python -m mlx_vlm generate \
  --model junafinity/Ornith-1.5-35B-A3B-uncensored-MLX-8bit \
  --prompt "Your prompt here" \
  --max-tokens 512

# image + text
python -m mlx_vlm generate \
  --model junafinity/Ornith-1.5-35B-A3B-uncensored-MLX-8bit \
  --prompt "Describe this image." \
  --image photo.jpg \
  --max-tokens 512
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("junafinity/Ornith-1.5-35B-A3B-uncensored-MLX-8bit")
config = model.config

prompt = apply_chat_template(processor, config, "Your prompt here", num_images=0)
print(generate(model, processor, prompt, max_tokens=512, verbose=False))
```

### LM Studio

Search `junafinity/Ornith-1.5-35B-A3B-uncensored-MLX-8bit` and import as an MLX model. No extra projector file is required (vision is inside the repo).

> ⚠️ `mlx-vlm` drops `mtp.*` during conversion. This MLX 35B build does **not** carry the MTP head. Use the GGUF-8bit sibling if you need MTP.

## Responsible use

Primary intended use is **red teaming and defensive cybersecurity research**. See the section of that name above.

This model has had safety guardrails **reduced or removed**. Do not expose it as a public endpoint without an independent moderation layer. You are responsible for compliance with the base model's license and acceptable-use policy, applicable law, and the terms of any platform you deploy on. Removing guardrails does not remove accountability.
