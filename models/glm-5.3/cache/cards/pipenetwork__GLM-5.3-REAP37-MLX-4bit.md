---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm_moe_dsa
- mixture-of-experts
- 4-bit
pipeline_tag: text-generation
library_name: mlx
---

# GLM-5.3-REAP37-MLX-4bit

MLX (Apple Silicon) build of [**GLM-5.3**](https://huggingface.co/zai-org/GLM-5.3) — 744B-parameter
`glm_moe_dsa` MoE (256 experts, top-8; MLA with DeepSeek-V3.2-style sparse attention) — quantized
to **4-bit**.

**These files are modified**: converted from the upstream **bfloat16** release
([GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)) to MLX and quantized; the architecture
is unchanged. The multi-token-prediction layer (78) is not included.

## Runtime — read this

This checkpoint bundles `glm_moe_dsa.py` (declared via `model_file`) and needs it:

```bash
pip install -U mlx-lm
mlx_lm.generate --model pipenetwork/GLM-5.3-REAP37-MLX-4bit --trust-remote-code --prompt "..." --max-tokens 300
```

mlx-lm's own `glm_moe_dsa` builds a lightning indexer on all 78 layers, but GLM-5.2/5.3 ship
indexer weights on 21 (`indexer_types`: the other 57 "shared" layers reuse the previous full layer's
top-k selection). A strict load of the release fails with 285 missing parameters; `mlx_lm.load`
loads leniently and leaves those 57 indexers at random initialisation. Prompts up to 2048 tokens are
unaffected (the indexer is bypassed below `index_topk`); beyond that, 57 layers attend to keys
chosen by random projections. The bundled runtime implements the schedule as the reference does,
plus the reference's fp32 indexer scores and router logits and the indexer LayerNorm epsilon.
Tiny-config parity against `transformers` 5.16 is **4e-7** with the sparse path live, cached decode
exact; strict loading of this checkpoint reports zero missing and zero unexpected tensors. Details and
tests: [https://github.com/PipeNetwork/glm53-mlx](https://github.com/PipeNetwork/glm53-mlx).

## Size and what is quantized

**0.0 GB** on disk. RAM: 384 GB-class.

| group | share of parameters | this build |
|---|---:|---|
| routed experts (`switch_mlp`, 75 layers × 256) | 724.8B (97.5%) | 4-bit, group 64 |
| attention (MLA), shared experts, dense layers 0–2, embeddings, `lm_head` | 18.4B (2.5%) | 4-bit, group 64 |
| lightning indexer (21 layers), MoE router + correction bias, norms | 0.3B | as stored (bf16 / fp32) |

Source precision: the FP8 release is a lossy derivative of the bf16 one (dequantized FP8 weights
differ from bf16 by up to 1.6e-2 on values of 0.46 — half an e4m3 step). The ladder row `fp8` is the FP8 release itself measured against bf16: its error is the floor any FP8-sourced build inherits.

## REAP pruning

This build keeps **161 of 256** routed experts per MoE layer (37% pruned; the 3 dense layers, attention, shared experts and the router are untouched), chosen by REAP saliency — mean `router_weight × ‖expert_output‖` over 65,536 calibration tokens (wikitext-2 *train*, ten languages of Wikipedia and code; checked for zero 32-gram overlap with the eval set). Kept experts carry 78.3% of the layers' saliency mass on average. Ranking the experts on two disjoint halves of the calibration set picks the same kept set 89.8% of the time. The pruning was applied to the already-quantized 4-bit build, which is exactly equivalent to pruning bf16 and requantizing (expert subsetting and affine groups are on different axes). Saliency retention is not a quality measure — the perplexity below is.

## Quality

Two measurements, because at 744B most of the ladder cannot be loaded on a 512 GB machine:

**Per-layer divergence vs bf16** (`scripts/eval_ladder.py`): every decoder layer run in bf16 and in
each recipe on identical inputs (16,384 tokens of wikitext-2), *teacher-forced* (each layer
sees bf16 inputs — isolates its own damage) and *free-running* (each recipe feeds itself — what
inference does). Relative L2 error of the layer output; lower is better.

| recipe | teacher-forced (mean over layers) | free-running (final layer) | cosine (final) |
|---|---:|---:|---:|
| 8bit | 0.00685 | 0.13119 | 0.98945 |
| 6bit | 0.01465 | 0.16736 | 0.98389 |
| 5bit | 0.02651 | 0.22521 | 0.97272 |
| 4bit | 0.05161 | 0.35740 | 0.93390 |
| mixed-4_8bit | 0.02524 | 0.24951 | 0.96710 |
| mixed-3_6bit | 0.05242 | 0.42380 | 0.90624 |
| fp8 | 0.01741 | 0.17321 | 0.98320 |

**Perplexity** on wikitext-2 (test), 288,627 tokens in 141 windows of 2048, for the builds
that fit this machine, scored on identical windows:

| build | size | perplexity [95% CI] |
|---|---:|---|
| [4bit](https://huggingface.co/pipenetwork/GLM-5.3-MLX-4bit) | 418.6 GB | 2.8636 [2.6681, 3.0714] |
| [mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-MLX-mixed-4_8bit) | 427.8 GB | 2.7420 [2.5533, 2.9477] |
| [mixed-3_6bit](https://huggingface.co/pipenetwork/GLM-5.3-MLX-mixed-3_6bit) | 332.6 GB | 3.0338 [2.8366, 3.2386] |
| [REAP25-4bit](https://huggingface.co/pipenetwork/GLM-5.3-REAP25-MLX-4bit) | 316.6 GB | 3.2872 [3.0703, 3.5184] |
| [REAP37-4bit](https://huggingface.co/pipenetwork/GLM-5.3-REAP37-MLX-4bit) | 267.2 GB | 3.8517 [3.6212, 4.0937] |
| [REAP50-4bit](https://huggingface.co/pipenetwork/GLM-5.3-REAP50-MLX-4bit) | 214.7 GB | 5.0295 [4.7571, 5.3137] |

**Recommendation.** For a 512 GB Mac, **mixed 4/8-bit** (427.7 GB): perplexity 2.7420, a paired 4.3% better than uniform 4-bit (ratio 0.9575 [0.9537, 0.9612], better on 98.6% of windows) for 9 GB more — the 2.5% of non-expert weights are worth their 8 bits, as on every model we have measured. Uniform 4-bit (418.6 GB) is the fallback when those 9 GB matter. **Mixed 3/6-bit** (332.6 GB) is the 384 GB-class option, at a real cost: 3.0338, +5.9% over 4-bit and +10.6% over mixed 4/8 — it leads the ladder for the first ten layers and then 3-bit expert damage compounds. Among the builds that cannot be run here, the ladder puts 8-bit closest to bfloat16 (free-running error 0.131), then 6-bit (0.167); the **upstream FP8 release scores 0.173, between 6-bit and 5-bit**, which is why these are converted from the bf16 release. 5-bit (0.225) sits just above mixed 4/8 (0.250) at 100 GB more.

Greedy generation (a collapse detector, not a ranking) is coherent on every published build.

## License

[GLM-5.3 license](LICENSE), as the upstream model. Port code: [https://github.com/PipeNetwork/glm53-mlx](https://github.com/PipeNetwork/glm53-mlx).
