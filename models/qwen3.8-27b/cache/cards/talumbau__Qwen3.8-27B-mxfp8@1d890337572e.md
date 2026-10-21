---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- mxfp8
- quantized
- compressed-tensors
- image-text-to-text
pipeline_tag: image-text-to-text
library_name: transformers
---

# Qwen3.8-27B — MXFP8 Quantization

This is an **MXFP8-quantized** derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced using [llm-compressor](https://github.com/vllm-project/llm-compressor) v0.13.0.

The quantized module set was chosen to match [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) **exactly** — same 407 Linear modules, same layers left in BF16 — so the two checkpoints differ only in scaling granularity and can be compared directly.

## Quantization details

| Property | Value |
|----------|-------|
| Format | OCP MX FP8 (E4M3) |
| Block size | 32 elements, along K |
| Scale dtype | uint8 (E8M0), stored as `[N, K/32]` |
| Weight quantization | Static, per 32-element group |
| Activation quantization | Dynamic, per 32-element group |
| Modules quantized | 407 |
| Checkpoint format | `compressed-tensors` (auto-detected by vLLM) |

### What is quantized

`Qwen3.8-27B` is a **vision-language model** with a **hybrid attention** text stack: of its 64 decoder layers, 48 use Mamba/GDN-style `linear_attention` and 16 use conventional `full_attention` (a 3:1 repeat, `full_attention_interval=4`).

| Group | Count |
|-------|-------|
| MLP `gate_proj` / `up_proj` / `down_proj` (all 64 layers) | 192 |
| `linear_attn` `in_proj_qkv` / `in_proj_z` / `out_proj` (48 layers) | 144 |
| `self_attn` `q_proj` / `k_proj` / `v_proj` / `o_proj` (16 layers) | 64 |
| MTP head Linears | 7 |
| **Total** | **407** |

### What is left in BF16

Matching the FP8 reference: the entire **vision tower** (`model.visual.*`), `lm_head`, `embed_tokens`, all RMSNorm weights, the SSM parameters (`conv1d`, `A_log`, `dt_bias`), the small `linear_attn.in_proj_a` / `in_proj_b` projections, and `mtp.fc`.

Every non-quantized tensor in this repo is **byte-identical** to the base checkpoint (792/792 verified) — the only difference from `Qwen/Qwen3.8-27B` is the 407 quantized modules.

> **Note on the MTP head:** `transformers` 5.14.1 declares `mtp_num_hidden_layers=1` but does not build an MTP module, so a naive `oneshot` run drops those weights entirely. The 7 MTP Linears here were quantized directly with the same MXFP8 math, verified bit-exact against llm-compressor's output before being written.

## Verification

Validated on an MI-300X. Weights were dequantized and compared against the BF16 original:

| Check | Result |
|-------|--------|
| Module set vs FP8 reference | 407/407, 0 extra, 0 missing |
| Weight / scale dtypes | `float8_e4m3fn` / `uint8`, 0 violations |
| Scale shapes `[N, K/32]` | 0 errors |
| Dequant relative error | 2.649% – 2.660% (mean 2.656%) |
| NaN / Inf | 0 |
| Non-quantized tensors byte-identical to base | 792 / 792 |
| CPU logit check vs BF16 | top-1 match, cosine 0.999773, KL 0.00059 |

> **Not verified:** MXFP8 *hardware* GEMM kernels. That requires gfx950; the validation host was gfx942. The checks above confirm the checkpoint is well-formed and numerically sound, not that the fast path executes correctly.

## Hardware requirements

MXFP8 hardware acceleration requires **AMD MI350X / MI355X (CDNA4, gfx950)** with ROCm ≥ 7.2.0, or NVIDIA Blackwell (SM100+).

> This checkpoint will NOT run correctly on AMD MI300X (gfx942) or NVIDIA Hopper (H100/H200). Those GPUs use different FP8 encodings (FNUZ and E4M3FNUZ) that are incompatible with the OCP MX format stored here.

## Usage with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="talumbau/Qwen3.8-27B-mxfp8",
    # vLLM auto-detects compressed-tensors format — no --quantization flag needed
)

outputs = llm.generate(
    ["Tell me about AMD MI350X"],
    SamplingParams(max_tokens=200, temperature=0.7),
)
print(outputs[0].outputs[0].text)
```

Or from the command line:
```bash
vllm serve talumbau/Qwen3.8-27B-mxfp8
```

## Comparison variants

This checkpoint is part of a format study. The three differ in *scaling granularity*, which is the interesting axis for GEMM kernel work:

| Variant | Repo | Weight scaling | Activation scaling |
|---------|------|----------------|--------------------|
| bf16 (original) | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | — | — |
| fp8 (DeepSeek-style block) | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 128×128 tile, BF16 scales | dynamic, 1×128 group |
| **mxfp8 (this repo)** | `talumbau/Qwen3.8-27B-mxfp8` | 32-element group, E8M0 scales | dynamic, 32-element group |

Per-channel/per-token scaling is a rank-1 epilogue operation; 128×128 block scaling requires rescaling inside the K-loop; MXFP8's 32-element E8M0 groups map onto the gfx950 scaled-MFMA path. Same model, three different GEMM kernel shapes.

## License

Apache 2.0, inherited from the base model [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
