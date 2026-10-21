---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqe
- imatrix
- quantized
- qwen4_exp
- mtp
---

# Qwen3.8-Flash-Next-iq5-iq8-mixed-mlx

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding (PLE) and a 4B multi-token prediction (MTP) module, pairing Gated DeltaNet with Qwen Sparse Attention (QSA). It includes the language model, the n-gram embedding, the vision encoder and the MTP head.

Quantized with [oQ](https://github.com/jundot/omlx) (oMLX 0.7.1) streaming mixed-precision quantization with **oQe imatrix weighting** (importance-matrix-weighted group quantization plus a weighted least-squares refit of every scale/bias, aka "Improved oQe"). All routed-expert tensors are affine so they benefit from the imatrix; name coding: **iq5** = imatrix-weighted 5-bit experts, **iq8** = imatrix-weighted 8-bit experts.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 5-bit, group 128 (imatrix) | 100.663B | 55.92% | 61.52 GiB | 4.89 |
| Affine 8-bit, group 64 (imatrix) | 20.133B | 11.18% | 19.92 GiB | 8.49 |
| MXFP8, group 32 | 3.574B | 1.99% | 3.44 GiB | 8.26 |
| BF16 | 55.630B | 30.91% | 103.62 GiB | 16.00 |
| **Total** | **180.000B** | **100%** | **188.5 GiB (37 shards)** | **≈ 9.0** |

### Per-component recipe

| Component | Format |
|---|---|
| N-gram PLE embedding (51B) | BF16, kept whole (~95 GiB) |
| MTP head (`mtp.*`, incl. its experts/attention) | BF16 |
| Vision encoder (`model.visual.*`) | BF16 |
| `conv1d`, PLE key/value projections, routers (`mlp.gate`), `shared_expert_gate`, `A_log`, `dt_bias`, all norms, hyper-connection weights, `lm_head` | BF16 |
| `self_attn.{q,k,v,o}_proj`, `indexer.index_qk_proj` | MXFP8 g32 |
| `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}` | MXFP8 g32 |
| `embed_tokens`, `shared_expert.{gate,up,down}_proj` | MXFP8 g32 |
| Routed experts (`mlp.switch_mlp.{gate,up,down}_proj`), layers 0–39 | Affine 5-bit g128, imatrix-weighted |
| Routed experts, layers 40–47 | Affine 8-bit g64, imatrix-weighted |

## Imatrix calibration

- Streamed per-layer collection against the BF16 source: **1024 samples × 512 tokens** (adaptive expansion from 128), `oqe_code_multilingual` corpus
- 937 captured modules, expert-routing coverage 99.97% of 75,264 routed experts active
- Every affine group's scale/bias refit by weighted least squares under the imatrix (Improved oQe)
- Spot validation vs the non-imatrix sibling quant: needle-in-haystack retrieval at 64k, vision description, and multi-step math all pass; MTP draft acceptance ~89%

## Benchmarks (Apple M5 Max, 128 GB, omlx 0.7.1)

1000-token generations, temp 0, server-reported rates (`prompt tokens/s` / `generation tokens/s`):

| Context | Prefill | Decode |
|---|---|---|
| short (~100 tok) | 229 | 46.9 |
| 8k | 1131 | 43.4 |
| 64k | 1325 | 39.1 |

Varied job-suite wall times (chat / 2000-token gen / 8k & 64k prefill / mixed / 64k retrieval / vision / 4-way concurrent) are **5–15% faster** than the equivalent non-imatrix mixed quant (mainly from the affine-expert kernels for layers 40–47); 4×1000-token concurrency: 77 aggregate tok/s. Decode is unchanged by imatrix weighting itself.

## Usage (omlx)

```bash
brew install omlx   # 0.7.1 or newer
omlx serve          # port/API key per your settings.json
```

Recommended per-model settings in `~/.omlx/model_settings.json`:

```json
{
  "Qwen3.8-Flash-Next-iq5-iq8-mixed-mlx": {
    "qwen4_ple_ssd_offload": true,
    "mtp_enabled": true
  }
}
```

- With the 95 GiB n-gram table SSD-offloaded, the model runs at **~93.5 GB resident** (full-model footprint 197.9 GB — a resident load does not fit 128 GB machines).
- MTP speculative decoding engages automatically with `mtp_enabled` (adaptive policy; ~2.2–2.4 tok/cycle observed).
- Context window 262,144; the checkpoint ships the full vision encoder for image-text-to-text use.

## License

This checkpoint inherits the license of the source model. See the [original Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the full terms (Qwen Community License 1.0).
