---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm5_next
- mixture-of-experts
- 8-bit
pipeline_tag: image-text-to-text
library_name: mlx
---

# GLM-5.3-Flash-MLX-8bit

MLX (Apple Silicon) build of [**GLM-5.3-Flash**](https://huggingface.co/zai-org/GLM-5.3-Flash) — 320B-A18B
hybrid of 34 Kimi-Delta linear-attention layers and 11 DeepSeek-sparse-attention (NoPE MLA +
lightning indexer) layers with manifold-constrained hyper-connections — quantized to **8-bit**.

**These files are modified**: converted from the upstream bfloat16 release
([GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)) to MLX and quantized;
the architecture is unchanged. The multi-token-prediction layer (layer 45) is not included. The
vision tower is carried in bfloat16.

## Runtime

`glm5_next` landed in [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) `main` on 2026-08-26 (no
release carries it yet). Validating that port against `transformers` 5.16 at tiny scale found two
numerical bugs and two epsilon mismatches, which the runtime in
[https://github.com/PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx) fixes; parity is **1e-6** end to end, exact on cached decode.

| what | reference | mlx-vlm `main` | effect |
|---|---|---|---|
| `swiglu_limit` | gate clamped at 10, up at ±10, in every text MLP | no clamp anywhere in the text stack | formula mismatch on all 45 FFN blocks |
| mHC `base`/`scale` dtype | float32 | converter casts to bf16; the Metal kernel then reads `base` as float4 | `comb` mixing matrix off by ~0.5 on every layer of a converted checkpoint |
| MLA low-rank norm eps | `rms_norm_eps` = 1e-5 | 1e-6 | small |
| indexer LayerNorm eps | 1e-6 | 1e-5 | small |

This checkpoint keeps the mHC arrays and KDA decay parameters in float32 as stored, so it is
safe in either runtime; the clamp is a compute-path fix and needs the patched runtime:

```bash
git clone https://github.com/PipeNetwork/glm53-flash-mlx && cd glm53-flash-mlx && pip install -r requirements.txt
python scripts/smoke_generate.py /path/to/GLM-5.3-Flash-MLX-8bit
```
```python
from glm53_flash_mlx.load import load
model, processor = load("/path/to/GLM-5.3-Flash-MLX-8bit")
```

## Size and what is quantized

**334.1 GB** on disk (bfloat16 upstream: 642.7 GB).

| group | share of parameters | this build |
|---|---:|---|
| routed experts (`switch_mlp`, 42 layers × 288) | 304B (97%) | 8-bit, group 64 |
| KDA and MLA projections, shared experts, dense MLPs, embeddings, `lm_head` | ~9B | 8-bit, group 64 |
| lightning-indexer projections | 0.06B | 8-bit, group 64 |
| MoE router + correction bias, mHC arrays (fp32), KDA `A_log`/`dt_bias` (fp32), convolutions, norms | — | as stored |
| vision tower | 0.56B | bfloat16 |

## Quality

Perplexity on wikitext-2 (test), 288,627 tokens in 141 windows of 2048, every build scored
on **identical** windows through this runtime. The 643 GB bfloat16 model does not fit a 512 GB
machine, so the 8-bit build is the anchor (on every model we have measured, 8-bit has been
statistically indistinguishable from bfloat16). Per-window NLL differences against 8-bit,
bootstrapped over one shared index set (20,000 resamples):

| build | size | perplexity | ΔNLL/token vs 8-bit [95% CI] | windows worse |
|---|---:|---:|---|---:|
| [8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-8bit) | 334.1 GB | 3.4607 | — | — |
| [6bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-6bit) | 255.9 GB | 3.4646 | +0.0011 [−0.0017, +0.0038] | 89/141 |
| [mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit) | 181.9 GB | 3.5705 | +0.0312 [+0.0271, +0.0355] | 131/141 |
| [4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-4bit) | 177.6 GB | 3.7549 | +0.0816 [+0.0755, +0.0879] | 140/141 |
| [REAP25-mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP25-MLX-mixed-4_8bit) | 139.1 GB | 4.2249 | +0.1995 [+0.1657, +0.2377] | 139/141 |
| [REAP37-mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP37-MLX-mixed-4_8bit) | 118.3 GB | 4.8752 | +0.3427 [+0.2968, +0.3929] | 141/141 |
| [REAP50-mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit) | 96.3 GB | 6.0757 | +0.5628 [+0.5071, +0.6219] | 141/141 |
| [REAP25-4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP25-MLX-4bit) | — | 4.4361 | +0.2483 [+0.2135, +0.2873] | 141/141 |
| [REAP37-4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP37-MLX-4bit) | — | 5.1057 | +0.3889 [+0.3424, +0.4393] | 141/141 |
| [REAP50-4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP50-MLX-4bit) | — | 6.3840 | +0.6123 [+0.5552, +0.6722] | 141/141 |

Read the interval, not the point estimate; "windows worse" counts how many of the 141
windows the build lost outright.

Against the 8-bit anchor: 6-bit +0.1%, mixed 4/8-bit +3.2%, uniform 4-bit +8.5%. Routed experts are 97% of the parameters; the mixed build keeps the other ~9B (KDA and MLA projections, shared experts, dense layers, embeddings) at 8-bit for 4.4 GB more than uniform 4-bit.

Greedy generation (a collapse detector, not a ranking) is coherent on every published build.

## License

MIT, as the upstream model. Port code: [https://github.com/PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx).
