---
license: gemma
library_name: transformers
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
  - gemma4
  - moe
  - fp8
  - float8_e4m3fn
  - gb10
  - blackwell
  - quantized
  - self-quantized
language:
  - en
---

# Gemma 4 26B-A4B Instruct — FP8 (E4M3)

**By [AeyeOps](https://huggingface.co/aeyeops)** · Quantized from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)

## Why this checkpoint exists

Every publicly available FP8 checkpoint for Gemma 4 26B-A4B is broken.

Standard quantization tools — modelopt, llm-compressor, AutoFP8 — walk the
model's `named_modules()` looking for `nn.Linear` layers. But Gemma 4's MoE
experts don't use `nn.Linear`. They use a fused 3D `nn.Parameter` layout
inside the `Gemma4TextExperts` class, stacking all experts' weights into a
single `(num_experts, out_features, in_features)` tensor. The quantizers
don't recognize this, so they **silently skip the expert projections** —
which account for 91% of the model's parameters.

The result: checkpoints that appear to be FP8 but are mostly bf16, and
that fail to load or produce degraded output on `transformers ≥ 5.5`.

| Public checkpoint | Problem |
|---|---|
| LargitData/gemma-4-26B-A4B-it-FP8 | `compressed-tensors` format mismatch; experts not quantized |
| protoLabsAI/gemma-4-26B-A4B-it-FP8-dynamic | Dynamic quant config; experts not quantized |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | Same dynamic quant pattern; experts skipped |
| bg-digitalservices/gemma-4-26b-a4b-it-fp8 | `state_dict` key misalignment on transformers 5.x |

This checkpoint solves the problem by quantizing the 3D expert tensors
directly, using a custom per-(expert, output-channel) max-abs FP8 scheme
with bf16 scale buffers. A lightweight class-swap loader teaches
`from_pretrained` to materialize the FP8 parameters correctly.

## What's FP8 and what's not

This is a **mixed-precision** checkpoint. "FP8" refers specifically to the
MoE expert projections — not the whole model. Of 1073 tensors in the
checkpoint:

- **60 tensors are FP8** — the `gate_up_proj` and `down_proj` weights
  across all 30 MoE layers. These account for 91% of the model's
  parameters and are the primary memory bottleneck.
- **953 tensors remain bf16** — attention projections, embeddings, layer
  norms, the router, and the full vision encoder. Quantizing these would
  risk quality degradation for minimal memory savings.

The result is a 27 GB artifact that preserves bf16 precision everywhere
it matters for quality, while collapsing the expert weights to half size
where the capacity exists to absorb the rounding.

## Motivation

[AeyeOps](https://aeyeops.com) built this checkpoint while validating
[TurboQuant](https://pypi.org/project/turboquant/) — a 4-bit KV cache
compression library for long-context inference on memory-constrained
hardware. We needed a genuinely quantized Gemma 4 MoE checkpoint that
would fit within the 90 GB memory budget of an NVIDIA GB10 (128 GB unified
LPDDR5x). No public checkpoint met that bar, so we built our own.

The quantization pipeline, class-swap loader, and TurboQuant integration
all live in [**aeo-quant**](https://github.com/AeyeOps/aeo-quant) — an
open SDK from AeyeOps for running memory-constrained LLM inference on
heterogeneous hardware. See the [Quick start](#quick-start) below for a
working example that pairs this checkpoint with TurboQuant's 4-bit KV
cache.

---

## Quick start

> **Note:** Standard `AutoModelForCausalLM.from_pretrained(...)` will not
> work directly. The custom loader is required because transformers'
> built-in `Gemma4TextExperts` expects bf16 parameters, not FP8 bytes with
> scale buffers.

Install the loader from the [`aeo-quant`](https://github.com/AeyeOps/aeo-quant)
repository (not yet published to PyPI):

```bash
pip install "git+https://github.com/AeyeOps/aeo-quant.git#egg=aeo-quant[bridges]"
```

Then load the checkpoint:

```python
from aeo_quant.bridges.gemma4.loader import load_gemma4_fp8

model = load_gemma4_fp8("aeyeops/gemma-4-26b-a4b-it-fp8")
# Defaults: dtype=torch.bfloat16, device_map="cuda"
# Any from_pretrained kwargs are forwarded.
```

The loader is a context manager that temporarily swaps
`Gemma4TextExperts` → `Gemma4TextExpertsFP8` during construction, then
restores the original class on exit. No `trust_remote_code`, no runtime
monkey-patching. The standard state-dict path populates FP8 parameters
and bf16 scale buffers with zero missing/unexpected/mismatched entries.

### With TurboQuant KV cache compression

For long-context inference, pair this checkpoint with TurboQuant's 4-bit
KV cache to reduce memory pressure as conversation context grows:

```python
from turboquant import TurboQuantCache

cache = TurboQuantCache(bits=4)
outputs = model.generate(
    **inputs,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=4096,
)
```

---

## Model details

| | |
|---|---|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Architecture** | Gemma 4 — Mixture of Experts (26B total, 4B active) |
| **Quantization** | FP8 E4M3 (`torch.float8_e4m3fn`), per-expert-per-output-channel |
| **Weights quantized** | MoE expert projections (`gate_up_proj`, `down_proj`) — 60 tensors |
| **Weights preserved** | Attention, embeddings, norms, vision encoder (bf16) |
| **Artifact size** | ~27 GB (vs ~52 GB bf16) |
| **Format** | Sharded safetensors with `model.safetensors.index.json` |
| **Keys** | 1073 total (60 FP8 expert weights + 60 bf16 scales + 953 bf16 pass-through) |
| **Shards** | 6, ~5 GB each |
| **Quality** | 99.2% token-exact match vs native bf16 under greedy decoding |

---

## Quantization recipe

```python
def quantize_3d_to_fp8(weight_bf16):
    # weight_bf16: (num_experts, out, in) bfloat16
    max_abs = weight_bf16.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 448.0).clamp(min=1e-8).to(torch.bfloat16)
    weight_fp8 = (
        weight_bf16.to(torch.float32) / scale.to(torch.float32)
    ).to(torch.float8_e4m3fn)
    return weight_fp8, scale  # scale shape: (num_experts, out, 1)
```

- Per-(expert, output-channel) max-abs scale, no calibration data, no
  activation statistics, no stochastic rounding.
- `448.0` is the FP8-E4M3 max finite magnitude.
- bf16 scale buffers, flat-named (`gate_up_proj_scale`, `down_proj_scale`)
  to avoid collisions with the parameter namespace.
- Deterministic: same bf16 input → same FP8 output regardless of source.

The build is a shard-streaming pipeline that reads one input shard at a
time, quantizes fused 3D expert tensors in-flight, and passes every
non-expert tensor through unchanged. Peak CPU memory during build: ~16 GB
(the full bf16 model is never materialized in RAM).

---

## Validation

Tested on NVIDIA GB10 Max Pro (128 GB unified LPDDR5x, Blackwell SM121):

**Environment:** `torch 2.11+cu130`, `transformers 5.5.3`, `turboquant 0.2.0`

| Metric | FP8 (this checkpoint) | bf16 reference | Notes |
|---|---|---|---|
| Load time | 147.6 s | 246.9 s | FP8 is 40% faster to load |
| `torch_alloc` | 26.93 GB | 48.23 GB | **FP8 saves 21.3 GB** |
| `sys_used` peak | 38.83 GB | 59.60 GB | |
| tok/s (greedy) | 8.0 | 10.9 | bf16 is 36% faster (see below) |
| Load report | 0/0/0/0 | 0/0/0/0 | missing/unexpected/mismatched/errors |

### Token-level quality

Same prompt, same settings (`do_sample=False`, `max_new_tokens=128`,
`TurboQuantCache(bits=4)`):

```
Total mismatches over first 128 tokens: 1
Match rate: 99.2%
```

The single divergence is at token index 4 (`","` vs `" and"`), both
valid continuations of *"Here is a concise"*. The two sequences
immediately re-converge and the remaining 124 tokens are byte-identical.

### Throughput gap

The 8.0 vs 10.9 tok/s difference is the cost of dequantizing each active
expert's FP8 → bf16 on every MoE forward call. A native FP8 matmul path
on Blackwell (`torch._scaled_mm`) would close most of this gap; we have
not implemented that optimization in this release.

---

## Known limitations

1. **Custom loader required** — Standard `from_pretrained` will not handle
   the FP8 parameters without the class-swap loader.
2. **Per-call dequant overhead** — Each MoE forward pass reconstructs
   active experts' bf16 weights from FP8. Native FP8 matmul would be faster.
3. **Single-token greedy divergence vs bf16** — 1/128 tokens, with
   immediate re-convergence. Not suitable for tests requiring byte-exact
   equivalence.
4. **Long-context prefill memory** — An unrelated upstream
   `modeling_gemma4.py` issue causes memory pressure above ~16K tokens.
   This is not a quantization issue.
5. **No calibration data** — Scales are derived from weight magnitudes
   only, with no activation statistics or outlier handling beyond a
   1e-8 clamp.

## License

Inherits from the base model. Use of this checkpoint requires acceptance
of the [Gemma license](https://ai.google.dev/gemma/terms) at
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).
This repo does **not** re-grant any rights.

## Citation

If you reference this build, please also cite the base model per Google's
Gemma terms. This is a mechanical requantization, not an independent
model release.

## Changelog

- **2026-04-12** — Initial FP8 build via shard-streaming pipeline.
  Validated on GB10 Max Pro. 99.2% token match vs bf16 reference.
