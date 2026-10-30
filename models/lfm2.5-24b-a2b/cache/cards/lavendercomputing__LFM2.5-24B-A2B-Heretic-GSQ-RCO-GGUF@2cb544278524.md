---
license: other
license_name: lfm1.0
base_model:
  - LiquidAI/LFM2.5-24B-A2B
  - OS-Software/LFM2.5-24B-A2B-Uncensored-Heretic
library_name: gguf
formats:
  - gguf
pipeline_tag: text-generation
tags:
  - gguf
  - quantization
  - text-generation
  - moe
  - heretic-abliteration
---

# LFM2.5-24B-A2B — Heretic uncensor, GSQ-RCO-style GGUF

Per-tensor mixed-precision GGUF builds of the Heretic-abliterated
`LiquidAI/LFM2.5-24B-A2B`, made for llama.cpp.

## What this model is

`LiquidAI/LFM2.5-24B-A2B` is a ~24B mixture-of-experts text model in Liquid
AI's `lfm2_moe` family, activating only ~2B parameters per token. The GGUF
layout has 40 blocks: 38 MoE blocks with 64 experts each (1536×2048), 30
blocks with a short-convolution stack (`shortconv.in_proj/out_proj/conv`),
and 10 blocks with full attention (2048×512 K/V). Embeddings are tied
(there is no `output.weight`), the vocab is 128,000 tokens, and the context
length is 131,072. Routed experts are 95.7% of all parameters.

**License: LFM Open License v1.0** — local and research use is free;
commercial use is allowed for entities under the $10M annual revenue
threshold. Vendor card: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B

## Files

This is a text model; no `mmproj` file is needed.

| File | Size | BPW | Expert ladder (38 MoE blocks) |
|---|---|---|---|
| `quant/LFM2.5-24B-A2B-Heretic-GSQ-RCO-IQ3_XXS.gguf` | 8.39 GiB | 3.01 | 25 edge @ IQ3_XXS, 2 @ IQ2_S, 11 inner @ IQ2_XS |
| `quant/LFM2.5-24B-A2B-Heretic-GSQ-RCO-IQ3_S.gguf` | 9.78 GiB | 3.50 | 32 edge @ IQ3_S, 3 @ IQ3_XXS, 3 inner @ IQ2_S |

Upstream copies of the intermediate Q8_0 source and the importance matrix
are linked below.

## Source and provenance

The uncensor was produced with
[Heretic](https://github.com/p-e-w/heretic): a training-free abliteration
pass that estimates the model's refusal direction, projects it out of
selected weight matrices, and keeps trials under a KL bound with
behavioural checks.

```
vendor BF16
  → Heretic abliteration kept at BF16 (OS-Software)
  → Q8_0 GGUF (effectively lossless)
  → these IQ3_XXS / IQ3_S builds
```

- **BF16 uncensored master:** [`OS-Software/LFM2.5-24B-A2B-Uncensored-Heretic`](https://huggingface.co/OS-Software/LFM2.5-24B-A2B-Uncensored-Heretic)
- **Q8_0 source:** [`OS-Software/LFM2.5-24B-A2B-Uncensored-Heretic-GGUF` → `LFM2.5-24B-A2B-Uncensored-Heretic-Q8_0.gguf`](https://huggingface.co/OS-Software/LFM2.5-24B-A2B-Uncensored-Heretic-GGUF/blob/main/LFM2.5-24B-A2B-Uncensored-Heretic-Q8_0.gguf)
- **Importance matrix:** [`mradermacher/LFM2.5-24B-A2B-Uncensored-Heretic-i1-GGUF` → `LFM2.5-24B-A2B-Uncensored-Heretic.imatrix.gguf`](https://huggingface.co/mradermacher/LFM2.5-24B-A2B-Uncensored-Heretic-i1-GGUF/blob/main/LFM2.5-24B-A2B-Uncensored-Heretic.imatrix.gguf)
  — coverage: 516 of 517 significant 2-D matrices. The only gap is the tied
  `token_embd.weight`, which is pinned at Q8_0 and needs no rows.

## Quantization

"GSQ-RCO-style" means a format-compatible equivalent of the GSQ/RCO 3-bit
recipe; the published GSQ/RCO engines cannot emit GGUF.

The edge-weighted ladder runs on `ffn_(gate|up|down)_exps`; 95.7% of
parameters live in routed experts, so the experts carry the compression.
Fixed policies:

- `token_embd` Q8_0 (tied, so it also serves the output header);
- router `ffn_gate_inp` Q8_0;
- attention: `attn_q` / `attn_output` Q6_K, `attn_k` / `attn_v` Q8_0;
- shortconv: `in_proj` / `out_proj` Q6_K, the small 3×2048 `conv` Q8_0;
- dense MLP blocks 0 and 1: Q6_K;
- 1-D norms and biases: fp32.

Both builds were smoke-tested with CPU-only llama.cpp tooling: they load,
stay coherent, and produce reasoning output. The 8–11 t/s CPU number is a
smoke signal, not a serving number.

## Usage

```bash
llama-cli -m quant/LFM2.5-24B-A2B-Heretic-GSQ-RCO-IQ3_S.gguf
```

IQ3_S is the safer default; IQ3_XXS is the smaller option.

## Honest limits

- Heretic's behavioural test suite belongs to the community work, not this
  repository. These builds were checked for load, run, and coherence only.
- The IQ3 tiers inherit quantization loss. For strict quality comparisons,
  use OS-Software's Q6_K or BF16 path.
- This llama.cpp build supports the `lfm2_moe` architecture out of the box.
