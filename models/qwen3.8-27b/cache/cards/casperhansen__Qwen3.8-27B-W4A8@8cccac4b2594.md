---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - int4
  - int8
  - w4a8
  - integer-only
  - compressed-tensors
---

# Qwen3.8-27B-W4A8

An integer-only **W4A8** quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), meant for
accelerators that only do integer arithmetic (FPGAs, NPUs and similar). Every linear layer of the language model, except
the small MTP input projection `mtp.fc`, stores symmetric integer weights with **one scale per output channel** and no
zero point. The checkpoint stores no group-wise scales and no static activation scales. At inference time, each matmul
input is quantized to **INT8 with one dynamic scale per token** (see
[Activation quantization](#activation-quantization-online)).

The recipe is deliberately simple and needs no calibration data:

* **Weights:** round-to-nearest with a per-output-channel scale chosen by an MSE clipping search (a grid search over
  clipping ratios that minimises each row's weight-quantization error). Compared with plain absmax scaling, this cuts
  the INT4 weight error by about a third. No GPTQ, rotations, smoothing or training.
* **Activations:** dynamic symmetric absmax INT8 per token, computed online.
* **Source weights:** the official FP8 release ([Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)),
  dequantized to BF16 and then requantized.

Checkpoint size: **17 GB** (BF16: ~56 GB).

## What is quantized

| Tensors | Weights | Input activations |
|---|---|---|
| Linear attention (Gated DeltaNet): `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | INT4, per output channel | INT8, per token (dynamic) |
| Full attention: `q_proj` (incl. output gate), `k_proj`, `v_proj`, `o_proj` | INT4, per output channel | INT8, per token (dynamic) |
| MLP: `gate_proj`, `up_proj`, `down_proj` | INT4, per output channel | INT8, per token (dynamic) |
| `lm_head` | INT8, per output channel | INT8, per token (dynamic) |
| MTP layer (same modules as a decoder layer) | as above | as above |

These tensors are left in **BF16**, as in the base model:

* `embed_tokens`: a table lookup, not a matmul (quantize per row if needed);
* `linear_attn.conv1d`: the 4-tap depthwise causal convolution;
* all RMSNorm weights, `A_log`, `dt_bias`;
* `mtp.fc`;
* the vision encoder (`model.visual.*`).

The checkpoint doesn't prescribe a precision for non-matmul math (norms, SiLU, sigmoid gates, softmax, RoPE, the Gated
DeltaNet recurrent state) or for the KV cache. Choose fixed-point formats for those to suit the target hardware.

## Activation quantization (online)

Every quantized module in the table above runs as an integer matmul. Its **input** is quantized on the fly, separately
for each token (each row of the input) and for each module input, as follows:

```
x        : one token's input vector to the module, length K (hidden size or intermediate size)
s_x      = max(max_k |x_k|, 1e-12) / 127                       # FP32, computed at runtime, never stored
q_x[k]   = clamp(round(x_k / s_x), -127, 127)                   # INT8; round half to even; symmetric (no -128)
acc[o]   = sum_k q_x[k] * q_w[o, k]                             # exact integer accumulation, fits in INT32
y[o]     = acc[o] * s_x * s_w[o]                                # rescale (FP32 or a fixed-point multiplier)
```

* **Granularity:** one scale per token per module input. There are no per-channel or per-group activation scales, no
  static or calibrated scales, and no outlier handling (rotation or smoothing). The weight scale `s_w[o]` is per output
  channel, so it factors out of the sum and the whole reduction stays in integers.
* **Shared inputs:** modules that read the same vector see the same quantized input, so it can be quantized once and
  reused:
  * `q/k/v_proj`;
  * `in_proj_qkv/z/a/b`;
  * `gate_proj`/`up_proj` (both read the post-attention RMSNorm output).
* **Second-stage inputs are quantized again** with their own per-token scale:
  * `down_proj` input: `silu(gate) * up`;
  * `out_proj` input (Gated DeltaNet): the gated RMSNorm output;
  * `o_proj` input: the attention output after the sigmoid output gate;
  * `lm_head` input: the final RMSNorm output.
* **Accumulator range:** INT4 x INT8 with K ≤ 17408 (`down_proj`) gives |acc| ≤ 17408·8·127 ≈ 17.7 M. INT8 x INT8
  (`lm_head`, K = 5120) gives |acc| ≤ 5120·128·127 ≈ 83.2 M. Both fit in INT32.
* **Outputs:** matmul outputs (`y`) are higher precision. Everything between matmuls (residual stream, norms,
  activations, attention, the recurrent state) is outside this spec.

## Weight storage format

Standard [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) `pack-quantized` format, with three
tensors per quantized module:

| Tensor | dtype / shape | Meaning |
|---|---|---|
| `<module>.weight_packed` | int32 `[out, in·bits/32]` | integer codes packed along the input dimension |
| `<module>.weight_scale` | bfloat16 `[out, 1]` | per-output-channel scale `s_w` |
| `<module>.weight_shape` | int `[2]` | `[out, in]` |

* **Codes:** INT4 codes are in [-8, 7] and INT8 codes in [-128, 127]. The dequantized weight is `w = q_w * s_w`; there
  is no zero point.
* **Packing:** within each row, element `k` occupies bits `[k·b, k·b + b)` of the row's little-endian int32 words (LSB
  first), so INT4 has 8 codes per word and INT8 has 4. Each code is stored as the unsigned value `q + 2^(b-1)`. All
  input dimensions are multiples of 32, so no code crosses a word boundary.

The quantization config in `config.json` lists every module's scheme: `weights` (bits, `strategy: channel`,
`symmetric: true`) and `input_activations` (`num_bits: 8`, `type: int`, `strategy: token`, `dynamic: true`).

### Reference implementation (PyTorch)

```python
import torch

def unpack(packed: torch.Tensor, bits: int, shape) -> torch.Tensor:
    """weight_packed (int32) -> integer codes (int8), shape [out, in]."""
    out_f, in_f = (int(v) for v in shape)
    per = 32 // bits
    p = packed.to(torch.int64) & 0xFFFFFFFF                      # int32 words as unsigned
    shifts = torch.arange(per, dtype=torch.int64) * bits
    u = (p.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)          # element k of a row: bits [k*b, k*b+b), LSB first
    return (u.reshape(out_f, -1)[:, :in_f] - (1 << (bits - 1))).to(torch.int8)

def quantize_activation(x: torch.Tensor):
    """Online per-token INT8: one scale per row of x ([tokens, K])."""
    s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.round(x.float() / s).clamp(-127, 127).to(torch.int8)
    return q, s

def int_linear(x, q_w, s_w):
    """y = x @ W^T with INT8 activations, integer weights and an exact integer accumulation."""
    q_x, s_x = quantize_activation(x)
    acc = q_x.cpu().long() @ q_w.cpu().long().T                  # exact integer accumulation (fits in INT32)
    return acc.float() * s_x.cpu() * s_w.cpu().float().view(1, -1)

# e.g. for a module m with b = 4 (INT4) or 8 (INT8):
#   q_w = unpack(sd[f"{m}.weight_packed"], b, sd[f"{m}.weight_shape"].tolist())
#   y   = int_linear(x, q_w, sd[f"{m}.weight_scale"])
```

`unpack` matches compressed-tensors' own unpacker bit-for-bit. `int_linear` matches a floating-point
quantize-dequantize forward to within float rounding (~1e-7 relative).

## Quality

Measured against the official FP8 model, teacher-forced on 448 responses that the FP8 model generated (about 0.58 M
response tokens). The prompts are 64 each from MATH-500, HumanEval, OpenCodeReasoning, SWE-bench Lite, IFEval, MMLU-Pro
and UltraChat, at reasoning efforts low / medium / xhigh in roughly equal shares. Each model was scored in 4 passes. The
metric is **EAR** (expected acceptance rate): the mean over positions of Σ_k min(P_ref(k), Q(k)), i.e. 1 − total
variation distance over the reference's top-64 tokens. It is the probability that the two models emit the same next
token under an optimal coupling. Activations were quantized exactly as specified above, emulated by
quantize-dequantize.

| Model | Weights | Activations | EAR vs FP8 |
|---|---|---|---:|
| FP8 (official) vs itself, run-to-run noise | FP8 | FP8 | 99.75% |
| **This model** | INT4 (INT8 `lm_head`) | INT8 per token | **90.1%** (±0.5) |

Expect a noticeable quality drop compared with the FP8 or BF16 model: on average, the next-token distribution differs
from the FP8 model's by about 10% of its probability mass (total variation). The recipe trades accuracy for simplicity:
INT4 for every linear layer and INT8 per-token activations with no outlier handling. If accuracy matters more than size,
those are the two things to revisit (for example INT8 attention / linear-attention projections, or rotating
activations before quantizing them).

## Running on GPUs

This checkpoint targets integer hardware. GPU engines have no kernels for INT4/INT8 weights with dynamic INT8
activations. To evaluate on a GPU, dequantize the weights (`q_w * s_w`) and apply the quantize-dequantize of
`quantize_activation` to every quantized module's input. The quality numbers above were measured exactly that way.

## License

Apache 2.0, the same as the base model.
