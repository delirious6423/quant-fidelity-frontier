---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- awq
- int4
- quantization
- amd
- gfx1030
- vllm
---

# Qwen3.8-27B-AWQ-vd

Full-INT4 re-quantization of Qwen/Qwen3.8-27B using the
[awq-quant-recipe](https://github.com/ikantkode/awq-quant-recipe) pipeline.
Every linear projection quantized: self-attention (16 layers), gated delta-net
(48 layers), MLP (64 layers), and lm_head. Only `in_proj_a/b`, norms,
embeddings, and the vision tower remain fp16.

**16 GB** (vs ~55 GB fp16 base) — fits a single 32 GB GPU with room for KV cache.

## Measured performance

| Metric | Value |
|---|---|
| Single-stream decode (Radeon PRO V620 + tuned kernels) | **17.3 tok/s** |
| Model size | 16 GB |
| Quality gates | per-module rel-err ≤ 0.11 (ALS-fitted); Stage E PASS |

First-serve measurement with no per-shape kernel tuning applied — the 4B
equivalent went from ~45 to 97.9 tok/s through kernel optimization (see
[gfx1030-vllm-0.26](https://github.com/ikantkode/gfx1030-vllm-0.26)). Similar
tuning on the 27B's shapes is expected to improve throughput.

## Quantization details

- **Method**: AWQ INT4, group_size 128, asymmetric zero-point
- **Scale fitting**: alternating least-squares (ALS) per-group grid fitting
- **LN-fold fix**: the `(1+w)` norm-fold defect in the AutoAWQ fork is
  corrected (see [FINDINGS.md](https://github.com/ikantkode/awq-quant-recipe/blob/main/FINDINGS.md))
- **lm_head**: untied (`tie_word_embeddings: false`), INT4-quantized via ALS
- **`in_proj_a/b`**: fp16, exactly compensated with `1/s`

## Usage

```bash
vllm serve ikantkode/Qwen3.8-27B-AWQ-vd \
    --dtype float16 --max-model-len 8192 \
    --attention-backend ROCM_ATTN
```

Requires vLLM with ROCm support. On gfx1030 (Radeon PRO V620), use the
[deploy kit](https://github.com/ikantkode/gfx1030-vllm-0.26/tree/main/deploy)
for one-command setup.

## Lineage

- Base: `Qwen/Qwen3.8-27B` (Apache-2.0, bf16, ~55 GB)
- Recipe: `ikantkode/awq-quant-recipe` (8-step pipeline)
- First application of the recipe to a model larger than 4B — validates
  the pipeline's model-size-agnostic claim

## Related

- [4B equivalent](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd): 97.9 tok/s
- [Quantization recipe](https://github.com/ikantkode/awq-quant-recipe)
- [Serving kernels](https://github.com/ikantkode/gfx1030-vllm-0.26)
- [Docker image](https://hub.docker.com/r/exaroot/gfx1030-vllm-0.26)
