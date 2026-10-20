---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags:
  - gguf
  - quantized
  - apex
  - moe
  - mixture-of-experts
  - gemma4
  - vlm
  - vision
---





<!-- apex-banner-v2 -->
<div style="background-color: #f59e0b; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
<h2 style="color: white; margin: 0 0 10px 0;">⚡ Each donation = another big MoE quantized</h2>
<p style="font-size: 18px; margin: 0 0 15px 0;">I host <b>25+ free APEX MoE quantizations</b> as independent research. My only local hardware is an <b>NVIDIA DGX Spark</b> (122 GB unified memory), enough for ~30-50B-class MoEs, but <b>bigger ones (200B+) require rented compute</b> on H100/H200/Blackwell, typically $20-100 per quant.<br>If APEX quants are useful to you, your support directly funds those bigger runs.</p>
<p style="font-size: 20px; margin: 0;">
<a href="https://www.patreon.com/cw/mudler" style="color: white; text-decoration: underline;">🎉 Patreon (Monthly)</a> &nbsp;|&nbsp;
<a href="https://www.buymeacoffee.com/mudler" style="color: white; text-decoration: underline;">☕ Buy Me a Coffee</a> &nbsp;|&nbsp;
<a href="https://github.com/sponsors/mudler" style="color: white; text-decoration: underline;">⭐ GitHub Sponsors</a>
</p>
<p style="font-size: 14px; margin: 10px 0 0 0; opacity: 0.9;">💚 Big thanks to Hugging Face for generously donating additional storage, much appreciated.</p>
</div>

# Gemma 4 26B-A4B APEX GGUF

**APEX (Adaptive Precision for EXpert Models)** quantizations of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** | [APEX Project](https://github.com/mudler/apex-quant) | [Technical Report](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.pdf)

## Benchmark Results

Benchmarks coming soon (re-quantized with llama.cpp b8664 including Gemma 4 tokenizer and logit softcapping fixes). For reference APEX benchmarks on the Qwen3.5-35B-A3B architecture, see [mudler/Qwen3.5-35B-A3B-APEX-GGUF](https://huggingface.co/mudler/Qwen3.5-35B-A3B-APEX-GGUF).

## Available Files

| File | Profile | Size | Best For |
|------|---------|------|----------|
| gemma-4-26B-A4B-APEX-I-Balanced.gguf | I-Balanced | 19 GB | Best overall quality/size ratio |
| gemma-4-26B-A4B-APEX-I-Quality.gguf | I-Quality | 20 GB | Highest quality with imatrix |
| gemma-4-26B-A4B-APEX-Quality.gguf | Quality | 20 GB | Highest quality standard |
| gemma-4-26B-A4B-APEX-Balanced.gguf | Balanced | 19 GB | General purpose |
| gemma-4-26B-A4B-APEX-I-Compact.gguf | I-Compact | 15 GB | Consumer GPUs, best quality/size |
| gemma-4-26B-A4B-APEX-Compact.gguf | Compact | 15 GB | Consumer GPUs |
| gemma-4-26B-A4B-APEX-I-Mini.gguf | I-Mini | 13 GB | Smallest viable, fastest inference |
| mmproj.gguf | Vision projector | 1.2 GB | Required for image understanding |

## What is APEX?

APEX is a quantization strategy for Mixture-of-Experts (MoE) models. It classifies tensors by role (routed expert, shared expert, attention) and applies a layer-wise precision gradient -- edge layers get higher precision, middle layers get more aggressive compression. I-variants use diverse imatrix calibration (chat, code, reasoning, tool-calling, agentic traces, Wikipedia).

See the [APEX project](https://github.com/mudler/apex-quant) for full details, technical report, and scripts.

## Architecture

- **Model**: Gemma 4 26B-A4B (google/gemma-4-26B-A4B-it)
- **Layers**: 30
- **Experts**: 128 routed (8 active per token)
- **Total Parameters**: 26B
- **Active Parameters**: ~4B per token
- **Vision**: Built-in vision encoder (mmproj included)
- **APEX Config**: 5+5 symmetric edge gradient across 30 layers
- **Calibration**: v1.3 diverse dataset (chat, code, reasoning, multilingual, tool-calling, Wikipedia)
- **llama.cpp**: Built with b8664 (includes Gemma 4 tokenizer fix, logit softcapping, newline split)

## Run with LocalAI

```bash
local-ai run mudler/gemma-4-26B-A4B-it-APEX-GGUF@gemma-4-26B-A4B-APEX-I-Balanced.gguf
```

## Credits

APEX is brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team. Developed through human-driven, AI-assisted research. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp).
