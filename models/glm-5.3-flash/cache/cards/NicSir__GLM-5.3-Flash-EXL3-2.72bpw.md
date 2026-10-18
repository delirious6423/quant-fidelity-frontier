---
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
---

# GLM-5.3-Flash - EXL3 2.72bpw (local quantization)

> EXL3 quantization of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).

## Motivation

GLM-5.3-Flash is one of my favorite models. On 128GB RAM I use Unsloth's Q3-XXS quant, but it barely fits, so after hearing about the benefits of the EXL3 format, I quantized the model to try to match or improve on the GGUF's quality while leaving some headroom on my HW. I'm uploading it in case someone else finds it useful.

## Base model

- Model: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (`Glm5NextForConditionalGeneration`)
- 320B total / 18B active, 45 layers + 1 MTP layer (45), MoE with 288 experts + shared expert, multimodal text+vision.
- See the original model card at the link above for full details, benchmarks and license.

## Quantization recipe

- `quant_method: exl3, version: 1.4.6, bits: 2.72, head_bits: 6`
- `calibration: default (rows 250, cols 2048), out_scales: always, codebook: mul1`
- Size: ~107GB in 14 safetensors shards (`total_size 113957092908`).
- In EXL3 `bits = trellis_depth / 16`: `32=2-bit, 48=3-bit, 64=4-bit, 80=5-bit, 96=6-bit`.

## Build command

```bash
convert.py -i input_model -o output_model -w working_dir -b 2.70 -vb 16 -hb 6 -hq
```

- `-b 2.70`: target average bits per weight
- `-vb 16`: keep vision tower at 16-bit (not quantized)
- `-hb 6`: attention heads / high-precision layers at 6-bit
- `-hq`: high-quality quantization pass
- Calibration: default
- Final measured average: `2.72 bpw`.

## Bits per layer

| Layers | Module | Bits |
|---|---|---|
| 0-2 | `mlp.down / gate / up` (dense) | 3 |
| linear-attention layers (0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44) | `self_attn.qkv_proj, o_proj` | 4 |
| 3-15 | `mlp.experts.* up / gate / down` | 3 |
| 16-28 | `mlp.experts.* up / gate / down` | 2 |
| 29-44 | `mlp.experts.* up / gate / down` | 3 |
| 3-44 | `mlp.shared_experts down / gate / up` | 4 |
| 3,7,11,15,19,23,27,31,35,39,43 (sparse attention) | `q_a_proj, q_b_proj, kv_a_proj_with_mqa, indexer.wq_b` | 4 |
| 45 (MTP / NextN predict layer) | `experts, shared_experts, q_a / q_b / kv_a / wq_b / o_proj` | 6 |
| 45 | `eh_proj` | 5 |
| - | `lm_head` | 6 |

A handful of tensors fall back to 4-bit inside 2/3-bit blocks and vice versa due to calibration rounding.

## Not quantized

`embed_tokens`, final `norm`, `input_layernorm`, `post_attention_layernorm`, `hc_*` (Manifold-Constrained Hyper-Connections), router `mlp.gate.*`, small linear-attention projections (`f_*/g_*/b_proj/conv1d/dt_bias/A_log/o_norm`), `indexer.*` (except `wq_b`), `kv/q_layernorm`, `kv_b_proj`, and the full `model.visual.*` tower, all kept at original precision.

## Usage

Recommended runtime is [TabbyAPI](https://github.com/theroyallab/tabbyAPI), the de-facto server for EXL3.

It can also be loaded with `transformers` + `exl3` support using the included `quantization_config.json`, `config.json`, `tokenizer*` and `chat_template.jinja`, or with an `exllamav3` backend.

## Evaluation

Not formally evaluated yet. I plan to run some benchmarks soon to compare it against Unsloth's quant.
