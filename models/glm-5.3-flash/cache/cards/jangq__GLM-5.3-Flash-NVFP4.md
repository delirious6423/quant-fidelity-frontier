---
base_model: zai-org/GLM-5.3-Flash
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- glm5_next
- glm-5.3-flash
- nvfp4
- fp4
- fp8
- moe
- compressed-tensors
- reasoning
- agentic
- blackwell
---

<div align="center">
  <h1>GLM-5.3-Flash — NVFP4 (JANGQ-AI)</h1>
  <b>~330B MoE (288 routed experts) · NVFP4 experts · FP8 MLA attention · Blackwell / GB10 · TP2</b>
</div>

---

> **Runtime status: not yet validated end-to-end.** This is a `compressed-tensors` mixed-precision bundle
> (NVFP4 experts + FP8 attention). vLLM ≥ 0.31 supports `Glm5NextForConditionalGeneration` and the
> compressed-tensors NVFP4 path on Blackwell, but this exact bundle has not yet been served and checked for
> coherent output. Until that is done, treat it as a candidate, not a release. Native MTP (layer 45) is **not**
> included — pair with an external drafter (DFlash2) if you want speculative decoding.

## What this is

An **NVFP4** requantization of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (from the
BF16 release), targeted at NVIDIA **Blackwell** (RTX 50 / GB10 / DGX Spark). Built with `llm-compressor` 0.14 /
`compressed-tensors` 0.19 / `transformers` 5.17.

| Component | Precision |
| :--- | :--- |
| Routed experts (42 MoE layers × 288 × gate/up/down) | **NVFP4** — E2M1, group 16, FP8-E4M3 block scales + FP32 global scale, **static calibrated input scales** |
| MLA attention `q_a/q_b/kv_a/o_proj` (layers 3,7,…,43) | **FP8** E4M3 per-channel weights, dynamic per-token activations |
| KDA linear attention (34 layers), DSA indexer, router gates, shared experts, dense MLP (layers 0–2), embeddings, `lm_head`, vision tower | **BF16** (unchanged) |

- **189.1 GB on disk (176.1 GiB)**, 10 shards — fits **TP2 across two 128 GB GB10 nodes** with ~40 GiB/node left for KV cache.
- Vision tower preserved in BF16 (multimodal input intact).
- Verified from the safetensors headers: 36,288 packed expert matrices (42 × 288 × 3), 44 FP8 attention matrices.

## Quality (teacher-forced, vs. the BF16 source)

40 held-out conversations (never seen in calibration; 16 long max-effort reasoning traces, 8 short reasoning,
8 cyber-security Q&A, 8 raw coding prompts), 52,143 scored tokens, per-token KL(BF16 ‖ NVFP4) and top-1 agreement:

| Slice | Tokens | KL mean | KL median | p95 | p99 | Top-1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **All** | 52,143 | **0.146** | **0.0063** | 0.35 | 3.43 | **91.6 %** |
| Long max-effort reasoning | 32,752 | 0.076 | 0.0049 | 0.13 | 1.35 | 93.8 % |
| Short reasoning | 8,059 | 0.045 | 0.0019 | 0.15 | 0.62 | 94.5 % |
| Cyber-security Q&A | 5,079 | 0.276 | 0.049 | 1.04 | 5.73 | 82.0 % |
| Raw coding prompts | 6,253 | 0.541 | 0.011 | 3.00 | 11.6 | 84.3 % |

Reading: the typical token is near-lossless and reasoning text is clean; the mean is dominated by a ~1–2 % tail of
tokens (mostly raw code / security prompt text) where quantization flips expert routing. **Caveat:** these numbers
were measured on the bundle's weights dequantized to BF16 — i.e. weight error only. The static NVFP4 activation
scales are not simulated here; a real w4a4 runtime adds some activation error on top.

## Calibration

RTN (min-max) NVFP4 with static activation-scale calibration; every routed expert saw every calibration token
(`moe_calibrate_all_experts`). 225 conversations / 156k tokens rendered through the model's own chat template at
`reasoning_effort` **max/high**: GLM-5.3-Flash-authored reasoning traces, wide-language coding, security,
agentic/tool-calling and general prompts. Recipe shipped as `recipe.yaml`.

## Recommended generation config

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="glm-5.3-flash",
    messages=[{"role": "user", "content": "Find the bug and fix it."}],
    temperature=1.0, top_p=0.95, max_tokens=16384,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "max"}},   # low | high | max (default max)
)
```

Upstream sampling: `temperature=1.0`, `top_p=0.95`. Reasoning effort is a three-tier control in the chat template;
`max` is the default when unspecified.

## Credits

Quantized and evaluated by **Jinho Jang** (eric@jangq.ai) · JANGQ-AI · 2026-10-11.
License: inherits the base model's license.
