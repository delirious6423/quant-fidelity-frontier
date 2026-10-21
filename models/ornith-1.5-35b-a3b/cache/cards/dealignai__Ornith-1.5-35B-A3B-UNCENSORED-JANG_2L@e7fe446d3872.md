---
language:
- en
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- apple-silicon
- uncensored
- abliterated
- crack
- jang
- jang-2l
- reasoning
- vision
- video
- agentic-coding
- moe
- harmbench
- mmlu
- ornith
---

<div align="center">
<a href="https://vmlx.net"><img src="vmlx-app.png" alt="vMLX — run JANG models on Apple Silicon" width="820" /></a>
<br/><strong>Built for vMLX</strong> — the MLX inference engine for Apple Silicon with mixed-precision JANG bundles, KV-cache quantization, and agentic tool calling.
<br/><sub>Free for macOS · <strong>vmlx.net</strong></sub>
</div>

<h3 align="center">⚡ All JANG models are meant to be run in <a href="https://vmlx.net">vMLX</a></h3>

---

<div align="center">
<img src="dealign_mascot.png" width="128" />

# Ornith 1.5 35B — UNCENSORED CRACK
### JANG_2L  ·  2-bit mixed-precision (smallest)

**Uncensored** · **Vision + Video** · **Reasoning on by default** · **Agentic coding** · **262K context** · **~16 GB**

<a href="https://ko-fi.com/dealignai"><img src="https://img.shields.io/badge/Ko--fi-Support-FF5E5B?logo=ko-fi&logoColor=white&style=for-the-badge" alt="Ko-fi"></a>
</div>

---

## What Is This?

[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — a 35.9B
Mixture-of-Experts vision-language model (40 layers, 256 routed experts, hybrid gated-delta +
full-attention backbone, 27-layer vision tower, native video) — **uncensored** and quantized to a
2-bit mixed-precision (smallest) MLX bundle for Apple Silicon.

Refusal behavior is removed at the weight level: the model follows instructions across task
categories instead of refusing, while keeping its coding ability, knowledge, reasoning, and vision
intact. No runtime hooks, no steering vectors — a standard MLX bundle.

## Results

Measured on this exact bundle. MMLU is the standard 57-subject benchmark in logit mode. HarmBench
compliance is coherence-gated (looping or template dumps do not count) and excludes
copyright-reproduction behaviors. KL divergence is measured against the **uncracked MXFP8**
reference on neutral held-out text — lower means closer to the original model's behavior.

| Metric | Value |
|---|---|
| **MMLU (57-subject)** | 78.5% (base 80.5%, -2.02) |
| **HarmBench compliance** | 99.6% (239/240) |
| **KL vs uncracked MXFP8** | 0.2222 nats (floor 0.0000) |
| **Size** | ~16 GB |

### MMLU by category — base vs uncensored

| Category | Base | Uncensored | Δ |
|---|---:|---:|---:|
| STEM | 74.2% | 72.1% | -2.1 |
| Humanities | 82.3% | 78.1% | -4.2 |
| Social Sciences | 88.3% | 87.9% | -0.4 |
| Other | 80.8% | 79.6% | -1.2 |
| **Overall (57 subj)** | **80.5%** | **78.5%** | **-2.02** |

Capability is preserved: the model stays within a few points of the base bundle at the same
quantization while refusals are removed.

## Modalities

| | |
|---|---|
| **Vision** | supported — pass images through the bundled processor |
| **Video** | supported (native video preprocessor) |
| **Reasoning** | on by default; toggle with `enable_thinking` |
| **Tool calling** | native XML / function schema |
| **Context** | 262,144 |

## Usage

Run with [vMLX](https://vmlx.net) (recommended — honors the per-module mixed-precision overrides)
or an MLX-VLM runtime with `qwen3_5_moe` support.

Recommended sampling (coding preset): **temperature 0.6, top_p 0.95, top_k 20**. A general preset
(temperature 1.0) is also stamped in `jang_config.json`. Stop tokens
`eos_token_id = [248046, 248044]`.

```python
{
  "model": "dealignai/Ornith-1.5-35B-A3B-JANG_2L-UNCENSORED-CRACK",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.6, "top_p": 0.95, "top_k": 20,
  "enable_thinking": true
}
```

## Support dealignai

**[Support us on Ko-fi](https://ko-fi.com/dealignai)** · [X @dealignai](https://x.com/dealignai) · [dealign.ai](https://dealign.ai)

<div align="center"><img src="dealign_logo.png" alt="dealign.ai" width="200"/></div>

---

## ⚠️ Disclaimer

This model has had its safety-refusal behavior removed for research purposes. It will follow
instructions across all categories without refusing. You are solely responsible for how you use it
and for complying with all applicable laws. Published for AI-safety research and authorized
security testing.
