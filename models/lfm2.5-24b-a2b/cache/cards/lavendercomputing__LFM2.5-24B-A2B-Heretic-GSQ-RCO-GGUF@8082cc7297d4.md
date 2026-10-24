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

# LFM2.5-24B-A2B — Heretic uncensor, GSQ-RCO-style quantization

A ~24B mixture-of-experts text model from Liquid AI. It activates only about 2B parameters per token. CPU smoke tests measured 8–11 t/s using 16 threads.

## License

LFM Open License v1.0. Local and research use is free. Commercial use is allowed for entities under the $10M annual revenue threshold.

Vendor card: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B

## Files

| File | Size | BPW | Notes |
|---|---:|---:|---|
| `LFM2.5-24B-A2B-Heretic-GSQ-RCO-IQ3_XXS.gguf` | 8.39 GiB | 3.01 | smaller edge-weighted ladder |
| `LFM2.5-24B-A2B-Heretic-GSQ-RCO-IQ3_S.gguf` | 9.78 GiB | 3.50 | default low-bit build |

This is a text model. No mmproj file is needed.

## Architecture

Structure from the GGUF:

- 40 blocks total
- 30 blocks with short-convolution stacks
- 10 blocks with full attention
- 128,000-token vocabulary
- 131,072-token context
- tied embeddings, no separate `output.weight`

Routed experts are 95.7% of all parameters. The quantization ladder runs on `ffn_(gate|up|down)_exps`.

## Uncensor source

`OS-Software/LFM2.5-24B-A2B-Uncensored-Heretic` is the source uncensor. Heretic is training-free. It estimates the refusal direction, projects it out of selected weights, and keeps trials under a KL bound with behavioral checks.

The OS-Software repository publishes a BF16 master and its own GGUF ladder. This build uses their Q8_0 GGUF as the input source.

Provenance:

```text
vendor BF16
  -> Heretic abliteration kept at BF16
  -> OS-Software Q8_0 GGUF
  -> this repo's IQ3_XXS / IQ3_S ladder
```

Q8_0 is effectively one benign step away from full precision. For absolute reference quality, quantize directly from the BF16 master.

## Importance matrix

`models/lfm25/source/LFM2.5-24B-A2B-Uncensored-Heretic.imatrix.gguf`

Coverage is verified across 516 of 517 significant 2-D matrices. The only gap is tied `token_embd.weight`, which is pinned at Q8_0 and does not need imatrix rows.

## Quantization plan

Planner: `scripts/plan_alloc_lfm.py`, the edge-weighted MoE solver retuned for this inventory.

| File | BPW | Expert ladder |
|---|---:|---|
| IQ3_XXS | 3.01 | 25 edge blocks at IQ3_XXS, 2 at IQ2_S, 11 inner blocks at IQ2_XS |
| IQ3_S | 3.50 | 32 edge blocks at IQ3_S, 3 at IQ3_XXS, 3 inner blocks at IQ2_S |

Fixed policies:

- `token_embd`: Q8_0, tied output role
- `ffn_gate_inp`: Q8_0
- attention: `attn_q` and `attn_output` Q6_K
- attention: `attn_k` and `attn_v` Q8_0
- shortconv: `in_proj` and `out_proj` Q6_K, tiny `conv` Q8_0
- dense MLP blocks 0 and 1: Q6_K
- 1-D norms and biases: fp32

GSQ-RCO here means a format-compatible equivalent. Literal GSQ-RCO cannot emit GGUF.

## Usage

```bash
llama-cli -m LFM2.5-24B-A2B-Heretic-GSQ-RCO-IQ3_S.gguf
```

IQ3_S is the safer default. IQ3_XXS is the smaller option.

Both builds were smoke-tested with CPU-only llama.cpp f3f31a7 tooling. They load, stay coherent, and produce reasoning output. Treat the 8–11 t/s CPU smoke number as a smoke signal, not a serving number.

## Reproducing

```bash
scripts/fetch-lfm25.sh
scripts/plan_alloc_lfm.py models/lfm25/source/LFM2.5-24B-A2B-Uncensored-Heretic-Q8_0.gguf 3.00
scripts/plan_alloc_lfm.py models/lfm25/source/LFM2.5-24B-A2B-Uncensored-Heretic-Q8_0.gguf 3.50
scripts/quantize-lfm25.sh
```

## Caveats

- Heretic's behavioral test suite belongs to the community work, not this repository. These builds were checked for load, run, and coherence only.
- IQ3 tiers inherit quantization loss. For strict comparisons, use OS-Software's Q6_K or BF16 path.
- This llama.cpp build supports `lfm2_moe` out of the box.