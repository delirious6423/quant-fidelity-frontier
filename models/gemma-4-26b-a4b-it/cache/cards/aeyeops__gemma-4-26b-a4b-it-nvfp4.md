---
license: gemma
library_name: transformers
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
  - gemma4
  - moe
  - nvfp4
  - fp4
  - e2m1
  - gb10
  - blackwell
  - sm121
  - quantized
  - self-quantized
language:
  - en
---

# Gemma 4 26B-A4B Instruct — NVFP4 (E2M1)

**By [AeyeOps](https://huggingface.co/aeyeops)** · Quantized from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)

## Why this checkpoint exists

Every publicly available NVFP4 checkpoint for Gemma 4 26B-A4B is missing or
broken in the same way our [FP8 sibling](https://huggingface.co/aeyeops/gemma-4-26b-a4b-it-fp8)
had to fix: Gemma 4's MoE experts don't use `nn.Linear`, so standard
quantization tools walk the model's `named_modules()` and silently skip the
expert projections — the 91% of parameters that matter most for memory.

This checkpoint solves the problem by quantizing the 3D expert tensors
directly with NVIDIA's microscaled FP4 format (E2M1 with two-level scaling:
FP8 per-block scales and FP32 per-tensor scales), then running native
`mma.sync.*.kind::mxf4nvf4` matmul on consumer Blackwell through a
custom Triton kernel. The checkpoint ships with the
[**aeo-quant**](https://github.com/AeyeOps/aeo-quant) bridge that wires
the storage format into `transformers.generate()` end-to-end.

## What's NVFP4 and what's not

This is a **mixed-precision** checkpoint. "NVFP4" refers specifically to
the MoE expert projections. Of 1133 tensors in the checkpoint:

- **180 NVFP4 tensors** — three buffers per expert projection across 30
  MoE layers × 2 projections (`gate_up_proj`, `down_proj`):
  - 60 packed `uint8` weight buffers (two E2M1 values per byte)
  - 60 `float8_e4m3fn` per-block scale buffers (block_size=16)
  - 60 `float32` per-tensor scale buffers
- **953 tensors remain bf16** — attention projections, embeddings, layer
  norms, the router, and the full vision encoder. Quantizing attention
  projections to FP4 was tested and rejected: 13% per-matmul RMS noise
  compounds across 30 layers and flips greedy-argmax in a 262K vocab.

The result is a **~19 GB artifact** that preserves bf16 precision where
argmax is sensitive, while collapsing the 91%-of-parameters expert
weights to a quarter of their bf16 size.

## Motivation

[AeyeOps](https://aeyeops.com) built this checkpoint as part of
[aeo-quant](https://github.com/AeyeOps/aeo-quant) — an open SDK for
running memory-constrained LLM inference on heterogeneous hardware.
NVFP4 is the further step beyond the FP8 sibling: it shaves another
~8 GB off resident memory and — critically — enables native FP4 matmul
through Triton's `tl.dot_scaled` path on consumer Blackwell (`sm_120`,
`sm_121`), eliminating the per-call FP8 dequant overhead that bottlenecked
the FP8 decode path.

The quantization pipeline, native Triton kernels (2D prefill + 3D
fused-experts decode with kernel-side expert gather), class-swap loader,
and TurboQuant-3bit KV cache integration all live in
[**aeo-quant**](https://github.com/AeyeOps/aeo-quant). See the
[Quick start](#quick-start) below for a working example.

---

## Quick start

> **Note:** Standard `AutoModelForCausalLM.from_pretrained(...)` will not
> work directly. The custom loader is required because transformers'
> built-in `Gemma4TextExperts` expects bf16 parameters, not packed
> FP4 bytes with two-level scale buffers.

> **Hardware requirement:** Consumer Blackwell (`sm_120` / `sm_121`) with
> Triton ≥ 3.5. Datacenter Blackwell (`sm_100`, B100/B200) uses a different
> FP4 MMA encoding and has not been validated. Hopper / Ada / older GPUs
> are not supported.

Install the loader from the [`aeo-quant`](https://github.com/AeyeOps/aeo-quant)
repository (not yet published to PyPI):

```bash
pip install "git+https://github.com/AeyeOps/aeo-quant.git#egg=aeo-quant[bridges]"
```

Then load the checkpoint:

```python
from aeo_quant.bridges.gemma4.loader import load_gemma4_nvfp4

model = load_gemma4_nvfp4("aeyeops/gemma-4-26b-a4b-it-nvfp4")
# Defaults: dtype=torch.bfloat16, device_map="cuda"
# Weights stay packed FP4 on GPU; native mxfp4 matmul via Triton.
```

Or use the format-agnostic dispatcher with `QUANT_FORMAT=nvfp4`:

```python
from aeo_quant.core.config import quant_env
from aeo_quant.bridges.gemma4.loader import load_gemma4

quant_format, checkpoint, kv_bits = quant_env()
model = load_gemma4(checkpoint, quant_format=quant_format)
```

`quant_env()` auto-sets `TRITON_OVERRIDE_ARCH=sm120` before Triton
compiles — consumer Blackwell uses the sm_120 FP4 MMA encoding even on
`sm_121` silicon. **If you bypass the bridge and call the loader or a
Triton kernel directly, set `TRITON_OVERRIDE_ARCH=sm120` in your
environment before importing torch/triton.**

### With TurboQuant 3-bit KV cache

For long-context inference, pair this checkpoint with TurboQuant's hybrid
sliding-window-aware KV cache:

```python
from aeo_quant.bridges.gemma4.cache import Gemma4HybridTurboQuantCache

cache = Gemma4HybridTurboQuantCache(bits=3, config=model.config)
outputs = model.generate(
    **inputs,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=4096,
)
```

The hybrid cache caps compressed storage on Gemma 4's 25 sliding-window
layers at `sliding_window − 1 − residual_len` tokens (895 with defaults),
while the 5 full-attention layers grow unbounded. At 16K–32K context, the
hybrid cache eliminates ~80% of the per-step dequant work the stock cache
would have spent on masked-out tokens.

---

## Model details

| | |
|---|---|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Architecture** | Gemma 4 — Mixture of Experts (26B total, 4B active) |
| **Quantization** | NVFP4 E2M1 with FP8 per-block scales (block_size=16) and FP32 per-tensor scale |
| **Weights quantized** | MoE expert projections (`gate_up_proj`, `down_proj`) — 60 tensors, 180 NVFP4 buffers total |
| **Weights preserved** | Attention, embeddings, norms, router, vision encoder (bf16) |
| **Artifact size** | ~19 GB (vs ~27 GB FP8, ~52 GB bf16) |
| **Format** | Sharded safetensors with `model.safetensors.index.json` |
| **Keys** | 1133 total (180 NVFP4 buffers + 953 bf16 pass-through) |
| **Shards** | 4, ~5 GB each |
| **Quality** | Byte-for-byte token match vs pinned NVFP4 parity baseline (0/50 and 0/300 mismatches) |

---

## Quantization recipe

```python
def quantize_3d_to_nvfp4(weight_bf16, block_size=16):
    # weight_bf16: (num_experts, out, in) bfloat16
    # Per-tensor fp32 scale for overall range
    tensor_max = weight_bf16.abs().amax()
    tensor_scale = (tensor_max / 6.0).to(torch.float32)  # E2M1 max = 6
    # Per-block fp8 scale within each (out, 16-in) block
    blocks = weight_bf16.view(*weight_bf16.shape[:-1], -1, block_size)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    block_scale = (block_max / tensor_scale).to(torch.float8_e4m3fn)
    # Normalize to E2M1 range, pack two values per uint8
    normalized = blocks / (block_scale.float() * tensor_scale)
    packed_uint8 = pack_e2m1(normalized)  # two E2M1 values per byte
    return packed_uint8, block_scale, tensor_scale
```

- **Two-level scaling:** FP8 per-block scales absorb local magnitude
  variation within each 16-element block; FP32 per-tensor scale handles
  overall range. Matches NVIDIA's NVFP4 spec.
- **Packing:** Two 4-bit E2M1 values per `uint8`. `gate_up_proj_weight`
  shape `(ne, 2*im, hd//2)`; `down_proj_weight` shape `(ne, hd, im//2)`.
- **Deterministic:** same bf16 input → same NVFP4 output regardless of
  source hardware.

The build is a shard-streaming pipeline (`examples/build_checkpoint_nvfp4.py`
in aeo-quant) that reads one input shard at a time, quantizes fused 3D
expert tensors in-flight, and passes every non-expert tensor through
unchanged. Peak CPU memory during build: ~18 GB.

---

## Kernel paths

The decode-path throughput comes from a native Triton kernel that drives
consumer Blackwell's FP4 MMA instruction (`mma.sync.*.kind::mxf4nvf4`)
directly. Two launchers ship:

- **`nvfp4_linear_prequantized` (2D prefill)** — per-expert
  activation quantize + FP4 matmul + alpha fold in one kernel. Used
  when `M > 1`.
- **`nvfp4_linear_3d_gather` (3D decode)** — batched kernel that takes
  the full `(E=128, ...)` expert-weight tensor and picks each selected
  expert via `expert_ids[pid_e]` indirection — eliminates 120 per-step
  launches + the gathered-weight tensor allocations that would otherwise
  be required. Used when `M == 1`.

The fused alpha (`a_tensor_scale * w_tensor_scale`) is a 0-D device
tensor loaded inside the kernel epilogue, not a `.item()` on the host —
this removes a mandatory `cudaStreamSynchronize` per expert matmul and
makes the decode path compatible with `torch.cuda.CUDAGraph` capture.

---

## Validation

Tested on NVIDIA GB10 Max Pro (128 GB unified LPDDR5x, Blackwell SM121):

**Environment:** `torch 2.11+cu130`, `transformers 5.5.3`, `turboquant 0.2.0`,
`triton ≥ 3.5`, `TRITON_OVERRIDE_ARCH=sm120` (auto-set by `quant_env()`)

| Metric | NVFP4 (this checkpoint) | FP8 sibling | bf16 reference |
|---|---|---|---|
| Load time | 121.1 s | 147.6 s | 246.9 s |
| `torch_alloc` | **17.49 GB** | 26.93 GB | 48.23 GB |
| Artifact size | **~19 GB** | ~27 GB | ~52 GB |
| Decode tok/s (parity prompt) | **18.74** | 8.0 | 10.9 |
| Load report | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |

The 18.74 tok/s number is the shipped v0.1.14 decode throughput on a quiet
box; under memory pressure (vLLM co-resident + active swap) measured
numbers drift to the 12–16 tok/s range. The realistic ceiling inside
`transformers.generate()` is 20–30 tok/s; that substrate is a hard cap.

### Token-level quality

Same prompt, same settings (`do_sample=False`, `max_new_tokens=50`,
`Gemma4HybridTurboQuantCache(bits=3)`):

```
Parity check (50 tokens):   0/50 mismatches  — byte-exact vs pinned baseline
Long parity (300 tokens):   0/300 mismatches — byte-exact vs pinned baseline
```

The NVFP4 arithmetic path is deterministic and bit-stable: every release
since v0.1.4 (kernel-side alpha fold in 0.1.12, compile wrap stripped in
0.1.13, kernel-side expert gather in 0.1.14) has been validated as
byte-for-byte identical to the prior release.

### Throughput story

The 8.0 → 18.74 tok/s jump vs FP8 is not free — it comes from
progressively eliminating the launch overhead that bottlenecked the FP8
path on GB10:

1. **Native FP4 matmul** (v0.1.4) — no per-call dequant to FP8.
2. **Hybrid sliding-window cache** (v0.1.10) — no dequant on masked-out
   KV positions.
3. **Kernel-side alpha fold** (v0.1.12) — no host sync per expert matmul.
4. **3D fused-experts kernel** (v0.1.7) — replaces Python expert loop.
5. **Kernel-side expert gather** (v0.1.14) — eliminates 120 per-step
   launches + gathered-weight allocations.

---

## Known limitations

1. **Consumer Blackwell only.** Validated on `sm_121` (GB10) with
   `TRITON_OVERRIDE_ARCH=sm120`. `sm_100` (B100, B200, H200) uses a
   different FP4 MMA encoding and has not been tested. Hopper / Ada /
   older GPUs do not have FP4 MMA.
2. **Custom loader required.** Standard `from_pretrained` will not handle
   NVFP4 parameters without the class-swap loader. There is no plan to
   upstream this to `transformers`.
3. **Attention is bf16.** Quantizing Q/K/V/O projections to NVFP4 was
   tested and rejected at 88% parity divergence — FP4 per-matmul noise
   (~13% RMS, cosine 0.991) compounds past greedy-argmax tolerance
   across 30 layers in a 262K-vocab decode. The full
   forensic is in the aeo-quant plan archive.
4. **`Gemma4TextConfig.num_kv_shared_layers = 0` gotcha.** Using HF's
   built-in `StaticCache` on Gemma 4 requires stripping this attribute
   at both instance and class scope to avoid a `layer_types[:0] = []`
   truncation. Not an issue with `Gemma4HybridTurboQuantCache`.
5. **No calibration data.** Scales derive from weight magnitudes only —
   no activation statistics, no stochastic rounding. The per-block
   FP8 scale absorbs most of the outlier variance that would have been
   an issue with a single per-channel scale.

## License

Inherits from the base model. Use of this checkpoint requires acceptance
of the [Gemma license](https://ai.google.dev/gemma/terms) at
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).
This repo does **not** re-grant any rights.

## Citation

If you reference this build, please also cite the base model per Google's
Gemma terms. This is a mechanical requantization plus a Triton kernel
targeting consumer Blackwell, not an independent model release.

## Changelog

- **2026-04-20** — Initial NVFP4 build. Native `mma.sync.*.kind::mxf4nvf4`
  matmul on `sm_121` via Triton. Kernel-side expert gather for MoE decode
  (v0.1.14). 18.74 tok/s decode on GB10 parity prompt, byte-for-byte
  parity vs pinned baseline.
