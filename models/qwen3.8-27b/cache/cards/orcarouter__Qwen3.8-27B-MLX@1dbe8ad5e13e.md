---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
language:
  - en
  - zh
tags:
  - qwen
  - qwen3.8
  - qwen3_5
  - mlx
  - apple-silicon
  - quantized
  - 4-bit
  - 8-bit
  - vision-language
  - image-text-to-text
  - multimodal
  - mtp
  - function-calling
  - reasoning
---

<div align="center">

<a href="https://www.orcarouter.ai" target="_blank">
  <img src="https://www.orcarouter.ai/orca-logo-classic.png" alt="OrcaRouter" width="110">
</a>

<h1>Qwen3.8-27B-MLX</h1>

<p><em>An MLX build of the official Qwen3.8-27B — 4 / 6 / 8-bit + MTP drafter for Apple Silicon</em></p>


<!-- orcacyber-promo:start -->
<table>
<tr>
<td width="150" style="vertical-align: middle;">
<div align="center"><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5"><img src="https://www.orcarouter.ai/orca-logo-classic.png" width="90" alt="OrcaCyber Zero 1.5"></a></div>
</td>
<td>
<p>🛡️ <strong>New from Orca &mdash; <a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5">OrcaCyber Zero 1.5</a></strong></p>
<p>Post-trained for cybersecurity: vulnerability analysis, vulnerability reproduction, red teaming &amp; offensive-security research. <strong>98.01% pass@1 on CyberGym Level&nbsp;1</strong> &middot; 1M context &middot; function calling &middot; OpenAI-compatible API.</p>
<p><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5"><img src="https://img.shields.io/badge/Try%20it%20on%20OrcaRouter%20%E2%86%92-FF6A00?style=for-the-badge&amp;labelColor=0B1F3A" alt="Try it on OrcaRouter"></a></p>
</td>
</tr>
<tr>
<td colspan="2">
<p>🔌 <strong>New from Orca &mdash; <a href="https://www.orcarouter.ai/solutions/orca-connect">Orca Connect</a></strong></p>
<p>Running this in LM Studio? <strong>Orca Connect</strong> puts the rig under your desk behind your usual OrcaRouter API key, next to 200+ hosted models, reachable from a laptop, a phone or a cloud agent. Outbound only: no port forwarding, no tunnel, no firewall rule. <strong>Requests your own hardware serves cost $0</strong>, and <code>orca-connect claude</code> points Claude Code at your rig for one session.</p>
<video src="https://huggingface.co/spaces/orcarouter/README/resolve/main/assets/orca-connect.mp4" controls autoplay muted loop playsinline width="100%"></video>
<p><a href="https://www.orcarouter.ai/solutions/orca-connect"><img src="https://img.shields.io/badge/Connect%20your%20rig%20%E2%86%92-FF6A00?style=for-the-badge&amp;labelColor=0B1F3A" alt="Connect your rig"></a></p>
</td>
</tr>
</table>
<!-- orcacyber-promo:end -->

<p>
<a href="https://www.orcarouter.ai"><img src="https://img.shields.io/badge/Website-orcarouter.ai-1E6FEB" alt="Website"></a>
<a href="https://www.orcarouter.ai/models"><img src="https://img.shields.io/badge/OrcaRouter-Model%20Catalog-2EA043" alt="Model Catalog"></a>
<a href="https://www.orcarouter.ai/models/qwen/qwen3.8-27b"><img src="https://img.shields.io/badge/OrcaRouter-Model%20Card-6F42C1" alt="Model Card"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-4C8BF5" alt="License"></a>
<img src="https://img.shields.io/badge/Format-MLX-00A67E" alt="MLX">
<img src="https://img.shields.io/badge/Quants-4%20|%206%20|%208--bit-FF8800" alt="Quants">
<img src="https://img.shields.io/badge/Vision-preserved%20BF16-9B59B6" alt="Vision">
<img src="https://img.shields.io/badge/MTP-drafter-2EA043" alt="MTP">
</p>

<p><strong>One Gateway. Every Model.</strong> — Route Smarter · Ship Safer · Spend Less.</p>

<p>
<a href="https://www.orcarouter.ai">Website</a> ·
<a href="https://www.orcarouter.ai/models">Model Catalog</a> ·
<a href="https://www.orcarouter.ai/models/qwen/qwen3.8-27b">Model API</a> ·
<a href="https://github.com/Continuum-AI-Corp">GitHub</a> ·
<a href="https://discord.gg/yAh6Tex6kx">Discord</a> ·
<a href="https://x.com/OrcaRouter">X</a>
</p>

</div>

---

> An **MLX** build of the official [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
> — a 27B dense hybrid-attention (Gated DeltaNet linear + full attention) native vision-language model
> with thinking control, tool-calling and an **MTP** head — quantized to **MLX** for **Apple Silicon**.
> Three precisions (**4 / 6 / 8-bit**, affine) as subfolders, the **4-bit** build also mirrored at
> the repo root (so `orcarouter/Qwen3.8-27B-MLX` loads directly in LM Studio), plus a separate **`mtp/`
> drafter** for speculative decoding. The **vision tower, norms and conv layers stay in BF16**; only
> language-model linear weights (incl. `embed_tokens` / `lm_head`) are quantized. Browse all models in
> the [OrcaRouter Model Catalog](https://www.orcarouter.ai/models); deployed as API
> [here](https://www.orcarouter.ai/models/qwen/qwen3.8-27b).

---

## Available quantizations

| Folder | Bits | Group size | Size | Min Mac RAM | Quality vs BF16 |
|---|---|---|---:|---|---|
| `8-bit/` | 8 | 32 | ~31 GB | 32–64 GB | **Near-lossless** — best quality |
| `6-bit/` | 6 | 32 | ~24 GB | 32 GB | Excellent balance |
| `4-bit/` | 4 | 64 | ~16 GB | 24 GB | Very good — **recommended default** |
| `mtp/`   | bf16 | — | ~0.9 GB | — | MTP drafter (speculative decoding) |

> **Repo root = `4-bit/`.** `--model orcarouter/Qwen3.8-27B-MLX` (no subfolder) resolves to 4-bit.

---

## Quality — KL-divergence vs BF16

Measured on this build: **KL divergence** and **Top-1 token agreement** of each quantized precision
against the full-precision **BF16** reference (wikitext-2, 1024 tokens, MLX CUDA backend). Lower KLD /
higher Top-1 = closer to full precision.

| Precision | Mean KLD | KLD p95 | Top-1 agreement |
|---|---:|---:|---:|
| **8-bit** | 0.00068 | 0.00233 | **98.34 %** |
| **6-bit** | 0.00216 | 0.00795 | **96.97 %** |
| **4-bit** | 0.02824 | 0.10857 | **92.68 %** |

**8-bit / 6-bit are near-lossless** (Top-1 ≈ 97–98 %); **4-bit stays very close** (Top-1 ≈ 93 %) — the
recommended default for everyday use.

---

## Multi-Token Prediction (MTP) — speculative decoding

Qwen3.8-27B has a native **MTP** head. In MLX it is loaded as a **separate drafter**: the main model
is loaded with MTP stripped, and the drafter is passed explicitly. The drafter lives in
[`mtp/`](./mtp) (`model_type: qwen3_5_mtp`, block_size 3) and works with **any** main precision.

```bash
hf download orcarouter/Qwen3.8-27B-MLX --include "6-bit/*" "mtp/*" --local-dir ./Qwen3.8-27B-MLX

python -m mlx_vlm generate \
    --model       ./Qwen3.8-27B-MLX/6-bit \
    --draft-model ./Qwen3.8-27B-MLX/mtp \
    --draft-kind mtp --draft-block-size 4 \
    --prompt "Explain quantum entanglement in one sentence." --max-tokens 256
```

**Requirements:** an mlx-vlm build with the `qwen3_5_mtp` drafter and `--draft-kind mtp` (mlx-vlm
`main`). MTP acceptance is lossless — greedy output is identical, just fewer forward passes.

---

## Usage (mlx-vlm, Apple Silicon)

```bash
pip install -U mlx-vlm    # needs mlx-vlm >= 0.6.13, mlx >= 0.32

hf download orcarouter/Qwen3.8-27B-MLX --include "4-bit/*" --local-dir ./Qwen3.8-27B-MLX

# text
python -m mlx_vlm generate --model ./Qwen3.8-27B-MLX/4-bit \
    --prompt "Explain quantum entanglement in one sentence." --max-tokens 256
# vision (image + text)
python -m mlx_vlm generate --model ./Qwen3.8-27B-MLX/4-bit \
    --image path/to/image.png --prompt "Describe this image." --max-tokens 256
# OpenAI-compatible server
python -m mlx_vlm server --model ./Qwen3.8-27B-MLX/4-bit --port 8080
```

On Apple Silicon the Metal backend is used automatically. (On a Linux **CUDA** backend, vision requires
`MLX_CUDA_USE_CUDNN_SDPA=0`; not needed on macOS.)

### LM Studio

Repo root is the 4-bit build. Two things: **turn off KV-cache quantization** (unsupported on this
vision arch — load fails otherwise), and **update the MLX runtime** (*Settings → Runtime*) — `qwen3_5`
support landed in mlx-vlm 0.6.x.

---

# Qwen3.8-27B

Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, Qwen3.8 is the most
capable generation in the Qwen open-model family to date. Built on the architectural foundation of
Qwen3.5, Qwen3.8-27B is a compact, deployment-friendly dense model: a native vision-language model that
understands images and videos, with flexible thinking control, designed to carry complex, multi-step
tasks through to completion with greater reliability.

## Qwen3.8 Highlights

- **Core Capabilities:** comprehensive improvements across coding, professional work, research, and
  long-horizon agentic tasks.
- **Agent Execution:** stronger autonomous planning and better handling of environment feedback.
- **Downstream Compatibility:** broader support for popular harnesses and development tools.
- **Flexible Thinking Control:** thinking on by default, disable per request; tune depth with
  `reasoning_effort`, retain reasoning context via `preserve_thinking`.
- **Vision-Language Understanding:** native image and video understanding, from STEM diagrams and
  documents to hour-scale videos.

## Model Overview

- **Type:** Causal Language Model with Vision Encoder
- **Parameters:** 27B · **Hidden:** 5120 · **Layers:** 64 · **Vocab:** 248,320 (padded)
- **Hidden layout:** 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
  - **Gated DeltaNet:** 48 V heads / 16 QK heads, head dim 128
  - **Gated Attention:** 24 Q heads / 4 KV heads, head dim 256, RoPE dim 64
  - **FFN intermediate:** 17,408
- **MTP (Multi-Token Prediction):** trained with multiple steps (shipped as the `mtp/` drafter)
- **Context length:** 262,144 native, extensible to 1,000,000 tokens

## Best Practices

- **Sampling — thinking mode:** `temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`
- **Sampling — non-thinking:** `temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5`
- **Output length:** allow generous `max-tokens`; give ≥ 2048 headroom so the final answer isn't
  truncated by the thinking budget.
- **Ultra-long context:** for total length beyond 262,144, use RoPE scaling (e.g. YaRN).

## Citation

```bibtex
@misc{qwen38,
    title  = {Qwen3.8: A New Bar for Coding and Cowork},
    url    = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026}
}
```

## License

**Apache 2.0**, inherited from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
Quantization does not change the underlying license obligations.
