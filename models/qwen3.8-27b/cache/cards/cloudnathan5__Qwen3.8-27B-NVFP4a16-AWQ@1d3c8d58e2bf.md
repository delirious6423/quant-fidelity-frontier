---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - nvfp4
  - w4a16
  - awq
  - compressed-tensors
  - llm-compressor
  - vllm
  - quantized
---

# Qwen3.8-27B — NVFP4 W4A16 (AWQ)

4-bit weight-only quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) using
**AWQ** calibration and the **NVFP4** weight format.

- **Weights:** NVFP4 (E2M1), group size 16, FP8-E4M3 group scales
- **Activations:** bf16 — weight-only, i.e. **W4A16**
- **Size:** 52 GB → 20.6 GB
- Vision + video towers, MTP head, and full multimodal support preserved

Benchmarks to follow.

## Usage

### vLLM

```bash
vllm serve cloudnathan5/Qwen3.8-27B-NVFP4a16-AWQ \
  --max-model-len 32768
```

With MTP speculative decoding (the MTP head is included, in bf16):

```bash
vllm serve cloudnathan5/Qwen3.8-27B-NVFP4a16-AWQ \
  --max-model-len 32768 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

Then query it like any OpenAI-compatible endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="cloudnathan5/Qwen3.8-27B-NVFP4a16-AWQ",
    messages=[{"role": "user", "content": "Explain NVFP4 in two sentences."}],
)
print(resp.choices[0].message.content)
```

### transformers

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer

model_id = "cloudnathan5/Qwen3.8-27B-NVFP4a16-AWQ"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype="bfloat16", device_map="cuda:0"
)
tok = AutoTokenizer.from_pretrained(model_id)
```

## Scheme

| | |
|---|---|
| Preset | `NVFP4A16` (compressed-tensors) |
| Weight format | NVFP4 / FP4 E2M1, 4-bit |
| Strategy | `tensor_group`, `group_size=16`, symmetric |
| Group scales | FP8-E4M3, plus a per-tensor FP32 global scale |
| Input activations | none — bf16 at runtime |
| Targets | `Linear` |

NVFP4 is NVIDIA's 4-bit float format. Each group of 16 weights gets its own FP8-E4M3
scale, so the representation adapts to local dynamic range far more finely than INT4
with one FP16 scale per group of 128. Keeping activations at bf16 (**A16**) means this
runs anywhere vLLM runs — Blackwell (sm_100/sm_120) simply gets the fastest kernels.

## What AWQ does

AWQ (*Activation-aware Weight Quantization*) starts from the observation that weight
channels are not equally important. A small fraction of channels carry
disproportionately large activation magnitudes, and quantization error in those
channels dominates the error in the layer's output. Quantizing every channel with the
same granularity spends precision where it isn't needed and starves the channels that
matter.

Rather than keep those channels in higher precision — which breaks the uniform layout
kernels depend on — AWQ rescales them. For each group it searches a per-channel scale
`s`, divides the incoming activation by `s` and multiplies the consuming weights by
`s`. The product is mathematically unchanged, but the salient weight channels are
scaled into a range the 4-bit grid represents more accurately. The scale is folded into
the preceding normalization layer, so inference cost is identical to a plain quant.

Here the smoothing scales are searched over a 20-point grid per mapping
(`n_grid: 20`), with `duo_scaling` enabled so the search accounts for both activation
and weight magnitudes rather than activations alone.

## Calibration data

| | |
|---|---|
| Dataset | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| Split | `train_sft` |
| Samples | 256 |
| Max sequence length | 2048 tokens |
| Shuffle seed | 42 |
| Preprocessing | chat template applied, then tokenized with `add_special_tokens=False` |

General multi-turn instruction-following chat data, matching how the model is actually
used. This is the same calibration set llm-compressor uses in its own AWQ examples and
its `awq_nvfp4` CI config.

Calibration is **text-only**, which is deliberate: the vision tower is excluded from
quantization entirely, so it needs no calibration, and every layer that *is* quantized
lives in the language model and is exercised by text. llm-compressor's own Qwen3-VL AWQ
example calibrates the same way.

Note that AWQ scales are tuned to the calibration distribution. If your workload is
heavily code, long-context, or non-English, recalibrating on in-domain samples will
likely serve you better than this checkpoint does.

## What is and isn't quantized

Qwen3.8-27B is a hybrid linear-attention multimodal model: 64 layers interleaving 48
gated-DeltaNet `linear_attention` layers with 16 `full_attention` layers in a 3:1
pattern, plus vision/video towers and an MTP head. The quantization is scoped to match.

**Quantized — 400 `Linear` layers:**

| Count | Module |
|---|---|
| 48 | `linear_attn.in_proj_qkv` |
| 48 | `linear_attn.in_proj_z` |
| 48 | `linear_attn.out_proj` |
| 64 each | `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |
| 16 each | `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` |

**Left in bf16, deliberately:**

- `linear_attn.in_proj_a`, `linear_attn.in_proj_b` — the SSM gating (alpha/beta)
  projections. Tiny, so quantizing them saves almost nothing, but they drive the
  recurrent state dynamics and 4-bit here degrades long-context behaviour badly.
- `model.visual.*` — vision tower. Small share of parameters, disproportionate quality
  cost.
- `mtp.*` — multi-token-prediction head, kept intact so speculative decoding works.
- `lm_head`, `embed_tokens` — standard practice; the 248K vocab makes these
  accuracy-critical.
- All norms, `conv1d`, `A_log`, `dt_bias` — not `nn.Linear`, excluded by construction.

`in_proj_a` / `in_proj_b` still participate in the AWQ **balance** groups even though
they are not quantized. Every consumer of a smoothed activation has to receive the
compensating scale, or the transform is no longer mathematically equivalent.

## Recipe

```yaml
default_stage:
  default_modifiers:
    AWQModifier:
      requires_calibration_data: true
      mappings:
      - smooth_layer: re:.*layers\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.input_layernorm$
        balance_layers: ['re:.*self_attn.q_proj$', 're:.*self_attn.k_proj$', 're:.*self_attn.v_proj$']
      - smooth_layer: re:.*layers\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30|32|33|34|36|37|38|40|41|42|44|45|46|48|49|50|52|53|54|56|57|58|60|61|62)\.input_layernorm$
        balance_layers: ['re:.*linear_attn.in_proj_qkv$', 're:.*linear_attn.in_proj_z$', 're:.*linear_attn.in_proj_b$', 're:.*linear_attn.in_proj_a$']
      - smooth_layer: re:.*post_attention_layernorm$
        balance_layers: ['re:.*gate_proj$', 're:.*up_proj$']
      - smooth_layer: re:.*up_proj$
        balance_layers: ['re:.*down_proj$']
      duo_scaling: true
      n_grid: 20
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, 're:model\.visual\..*', 're:mtp\..*', 're:.*embed_tokens.*',
               're:.*\.linear_attn\.in_proj_a$', 're:.*\.linear_attn\.in_proj_b$',
               're:.*\.conv1d$', 're:.*norm.*']
      scheme: NVFP4A16
```

The two `input_layernorm` mappings are split by layer index because the architecture is
hybrid: layers 3, 7, 11, … are `full_attention` and smooth into `q/k/v_proj`, while the
rest are `linear_attention` and smooth into the DeltaNet input projections. These were
generated by llm-compressor's dynamic hybrid-attention mapping builder.

## Notes for anyone quantizing this architecture

Three issues came up that aren't obvious, recorded in case they save someone time.

**1. `inspect.signature` on the DeltaNet forward.** transformers 5.14 decorates
`Qwen3_5GatedDeltaNet.forward` with an accelerate helper that omits `functools.wraps`,
so `inspect.signature(module.forward)` reports `(*args, **kwargs)`. llm-compressor's AWQ
hook caches parent inputs via `inspect.signature(...).bind(*args, **kwargs).arguments`,
which against that opaque signature collapses everything into one `kwargs` key. Replay
then fails with `TypeError: Qwen3_5GatedDeltaNet.forward() missing 1 required
positional argument: 'hidden_states'`. Restoring `__wrapped__` on the wrapper fixes it,
affecting introspection only.

**2. Fused NVFP4 global scales.** vLLM fuses the split linear-attention projections via
`packed_modules_mapping["in_proj_qkvz"] = ["in_proj_qkv", "in_proj_z"]`, and fused
members must share one global scale. llm-compressor's `FUSED_LAYER_NAMES` covers
`(q_proj, k_proj, v_proj)` and `(gate_proj, up_proj)` but not this Qwen3.5-specific
pair, so they were assigned independent scales across all 48 linear-attention layers.
Registering the pair before calibration fixes it.

**3. The MTP head disappears twice.** transformers' `Qwen3_5ForConditionalGeneration`
does not define the `mtp.*` submodule, so its 15 tensors are silently dropped on load
and never written out. Separately, llm-compressor resolves ignore patterns against
modules present on the loaded model, so `re:mtp\..*` matched nothing and no MTP entry
reached the saved `quantization_config.ignore` either — leaving vLLM to build NVFP4
linears for the head and fail on bf16 weights with `ValueError: There is no module or
parameter named 'fc.weight' in Qwen3_5MultiTokenPredictor`. vLLM's existing escape
hatches don't cover this: the `mtp.fc` workaround is gated on `modelopt_fp4`, and the
`dynamic` `"-:mtp"` path is GPTQ-only. Both the tensors and the ignore entries have to
be restored explicitly.

## Environment

- llm-compressor 0.13.0, compressed-tensors 0.18.0
- torch 2.13.0+cu132, transformers 5.14.1
- Calibrated on an NVIDIA RTX PRO 6000 Blackwell (sm_120, 96 GB)

## License

apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
