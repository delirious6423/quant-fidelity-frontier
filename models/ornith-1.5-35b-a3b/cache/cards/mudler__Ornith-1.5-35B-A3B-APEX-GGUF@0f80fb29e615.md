---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
  - gguf
  - quantized
  - apex
  - moe
  - mixture-of-experts
  - qwen3
  - vlm
  - vision
---

<!-- apex-banner-v2 -->
<div style="background-color: #f59e0b; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
<h2 style="color: white; margin: 0 0 10px 0;">⚡ Each donation = another big MoE quantized</h2>
<p style="font-size: 18px; margin: 0 0 15px 0;">I host <b>30+ free APEX MoE quantizations</b> as independent research. My only local hardware is an <b>NVIDIA DGX Spark</b> (122 GB unified memory), enough for ~30-50B-class MoEs, but <b>bigger ones (200B+) require rented compute</b> on H100/H200/Blackwell, typically $20-100 per quant.<br>If APEX quants are useful to you, your support directly funds those bigger runs.</p>
<p style="font-size: 20px; margin: 0;">
<a href="https://www.patreon.com/cw/mudler" style="color: white; text-decoration: underline;">🎉 Patreon (Monthly)</a> &nbsp;|&nbsp;
<a href="https://www.buymeacoffee.com/mudler" style="color: white; text-decoration: underline;">☕ Buy Me a Coffee</a> &nbsp;|&nbsp;
<a href="https://github.com/sponsors/mudler" style="color: white; text-decoration: underline;">⭐ GitHub Sponsors</a>
</p>
</div>

# Ornith-1.5-35B-A3B APEX GGUF

APEX quantizations of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** | [APEX Project](https://github.com/localai-org/apex-quant)

These are the standard quants. For versions that bundle the MTP draft head for speculative decoding, see [Ornith-1.5-35B-A3B-APEX-MTP-GGUF](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF).

## Files

| File | Size | For |
|---|---|---|
| Ornith-1.5-35B-A3B-APEX-Quality.gguf | 22.82 GB | highest quality |
| Ornith-1.5-35B-A3B-APEX-Balanced.gguf | 25.27 GB | general purpose |
| Ornith-1.5-35B-A3B-APEX-Compact.gguf | 16.54 GB | consumer GPUs |
| Ornith-1.5-35B-A3B-APEX-I-Mini.gguf | 13.47 GB | smallest, imatrix only |
| mmproj.gguf | 0.90 GB | vision projector, pair with any of the above |

`I-` files use an importance matrix built from diverse calibration data (chat, code, reasoning, tool-calling, agentic traces, Wikipedia). Quality, Balanced and Compact also ship without it.

## The model

Ornith-1.5-35B-A3B is a 36 B parameter Mixture-of-Experts model with 256 routed experts and 8 active per token, plus a shared expert. It has 40 layers with hybrid attention, interleaving three linear-attention layers per full-attention layer, and a vision tower.

## How APEX quantizes it

Routed experts are **89.6%** of the weights here but only 8 of 256 fire for any given token, so they tolerate lower precision than the parts every token passes through. APEX classifies each tensor by role and applies a layer-wise precision gradient: the first and last layers keep higher precision, middle layers compress harder, and the always-active shared expert is kept high.

Attention is only 3.6% of the weights on this model (2.8% linear, 0.8% full), so it is not where the size is and is not treated as a lever.

## Usage

```bash
# text
llama-cli -m Ornith-1.5-35B-A3B-APEX-Balanced.gguf -p "Your prompt" -ngl 99

# vision
llama-mtmd-cli -m Ornith-1.5-35B-A3B-APEX-Balanced.gguf --mmproj mmproj.gguf -ngl 99
```

Needs a recent llama.cpp with `qwen3_5_moe` support.

## Notes

Sizes and quantization recipes are published in the [APEX repository](https://github.com/localai-org/apex-quant). No throughput benchmarks were run on these files.
