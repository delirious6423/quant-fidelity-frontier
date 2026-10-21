---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
library_name: mlx
base_model:
- Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
- mlx
- omlx
- oq
- quantized
- mxfp8
- moe
- qwen4_exp
- mtp
language:
- en
- zh
---

# Qwen3.8-Flash-Next-oQ5-mixed-mtp

Mixed-precision [MLX](https://github.com/ml-explore/mlx) quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(335 GiB bf16) produced with the **oQ streaming exporter** from
[oMLX](https://github.com/omlx/omlx) 0.7.1rc1. The artifact is a standard MLX
quantized checkpoint: `config.json` carries a `quantization` dict with
per-tensor `bits / group_size / mode` entries.

**Quantized size: 187.9 GiB** (37 safetensors shards) vs 335 GiB bf16 — a
1.78× reduction with the largest quality-critical components left in bf16.

## Base model

Qwen3.8-Flash-Next is a 125B-parameter Mixture-of-Experts vision-language
model (6B activated per token) with an experimental architecture previewing
Qwen4:

- 48-layer hybrid layout: 36 × (Gated DeltaNet → MoE) + 12 × (Qwen Sparse
  Attention → MoE)
- 512 routed experts, top-10 per token, plus one shared expert per layer
- 51B-parameter n-gram (PLE) embedding (20M vocab, bigrams/trigrams) — kept
  **bf16** in this quant
- 4B-parameter MTP (multi-token prediction) head — kept **bf16** for
  speculative decoding

## Quantization recipe

| Components | Precision | GiB |
|---|---|---|
| N-gram PLE embedding (20M vocab × 48-layer table) | bf16 | 95.4 |
| MTP head (all tensors, incl. its experts) | bf16 | 4.9 |
| Vision tower | bf16 | 0.9 |
| Hyper-connections, inject gates, routers, `shared_expert_gate`, convs (`linear_attn.conv1d`, `ple.*`), `A_log`/`dt_bias`, norms, `lm_head` | bf16 | ~1.5 |
| Routed experts, layers 0–39 (`switch_mlp.{gate,up,down}_proj`) | 5-bit affine, group 128 | 61.5 |
| Routed experts, layers 40–47 | mxfp8, group 32 | 8.0 |
| `self_attn.*` + QSA indexer, `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`, `embed_tokens`, `shared_expert.{gate,up,down}_proj` | mxfp8, group 32 | ~14.8 |
| **Total** | | **187.9** |

688 tensors pass through in bf16, 529 tensors are quantized. The recipe was
applied as an explicit per-tensor predicate (no automatic bit allocation);
zero tensors fell through to the default rule.

## Usage

### oMLX (verified)

oMLX discovers the model from its model directory:

```bash
omlx serve --model-dir /path/to/models --port 8000
# or with the brew service; then:
curl -H "Authorization: Bearer $OMLX_API_KEY" \
     127.0.0.1:8008/v1/chat/completions \
     -d '{"model":"Qwen3.8-Flash-Next-oQ5-mixed-mtp","messages":[{"role":"user","content":"Hello"}]}'
```

**Memory notes** (128 GiB machine): enable `qwen4_ple_ssd_offload: true` for
this model id in `~/.omlx/model_settings.json` — the 95 GiB n-gram table is
SSD-offloaded, leaving ~104 GiB resident. With `mtp_enabled: true` the bf16
MTP head drives speculative decoding at 4 tokens/cycle (~91% acceptance
observed on an initial smoke test).

### mlx-lm / mlx-vlm

The checkpoint uses the standard MLX quantized-weight layout
(`QuantizedLinear`/`QuantizedEmbedding` weights with `.scales`/`.biases`
partners and per-tensor mode entries), so mlx-vlm 0.7.8+ (which ships the
`qwen4_exp` architecture) and newer mlx-lm builds can reconstruct it.

### Transformers / vLLM / SGLang

Not applicable — this is an MLX-only artifact. Use the upstream bf16
checkpoint for GPU frameworks.

## Conversion

Converted 2026-10-16 with `omlx.oq.quantize_oq_streaming` (oQ level 5,
bfloat16 scales/biases, `preserve_mtp=True`, `preserve_ngram_table=True`).
Streaming conversion peaked at ~8.5 GiB RSS and completed in ~3 minutes.

## License

Inherited from the upstream checkpoint:
[Qwen Community 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
