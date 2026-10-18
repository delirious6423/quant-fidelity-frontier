---
language:
- en
- zh
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
tags:
- gguf
- glm
- glm-5.3
- quantized
- mtp
- dgx-spark
---

# GLM-5.3-Flash Spark Q2XL MTP

**Spark** is a quality-first 2.80 BPW GGUF quant of
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash),
tuned to fit and run fully on a 128 GB NVIDIA DGX Spark / GB10 with native MTP
speculative decoding.

The eight model shards total 106,987.66 MiB (about 105 GiB). This build keeps
attention and other sensitive paths at higher precision while compressing the
large routed-expert tensors more aggressively:

- routed expert down projections: `IQ3_XXS`
- routed expert gate/up projections: `IQ2_XS`
- shared experts: `Q6_K`
- attention Q/K/V/output matrices: `Q8_0`
- norms, routing controls, recurrent/SSM controls, and MTP-sensitive tensors:
  retained as BF16/F32 where required
- calibration: importance-matrix guided

The result is not simply a uniform 2-bit model. Most storage savings come from
the repeated expert matrices; the comparatively small, quality-sensitive paths
receive substantially more precision.

## Benchmarks

All local results below were measured on one DGX Spark / GB10. Quality used one
sample per task, temperature 0, top-p 0.95, `reasoning_effort=low`, a 768-token
completion cap, and EvalPlus 0.3.1 with HumanEvalPlus dataset v0.1.10. This is a
concise local-coding profile, not the upstream model's default `max` reasoning
profile.

| Test | Result |
|---|---:|
| HumanEval pass@1 | **159/164 (97.0%)** |
| HumanEval+ Mini pass@1 | **153/164 (93.3%)** |
| 34-point coding microbenchmark | 29/34 (85.3%) |
| Generation failures | 0/164 |
| Full generation wall time | 1,025 s, one sequential slot |

Five sequential streaming coding completions, 512 output tokens each:

| Metric | Result |
|---|---:|
| Median decode | **22.504 tok/s** |
| Mean decode | 22.083 tok/s |
| Minimum decode | 17.176 tok/s |
| Median effective completion | 21.660 tok/s |
| Median TTFT | 0.807 s |
| First-trial TTFT | 0.930 s |
| Median native-MTP acceptance | 54.3% |

For comparison, the previously tested Unsloth UD-Q2_K_XL GLM quant reached
155/164 HumanEval, 148/164 HumanEval+ Mini, and 25.414 tok/s median. Spark traded
about 11% median decode speed for five additional HumanEval+ passes in this
evaluation.

## Tested runtime

The model was converted and served with the GLM5-Next fork of llama.cpp at
commit `ea37b8bb0` (`vcruz305/llama.cpp`). Native MTP must be enabled. The
tested server configuration was:

```bash
llama-server \
  -m GLM-5.3-Flash-Spark-Q2XL-MTP-00001-of-00008.gguf \
  --host 0.0.0.0 --port 11001 \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --cache-ram 512
```

The 512 MiB prompt-cache cap is intentional. The server image's 8 GiB default
left only about 1 GiB of memory headroom and caused a response-finalization
timeout during sustained non-streaming requests. With the cap, the model loaded
with about 9 GiB available and completed the full 164-task run cleanly.

The upstream model is natively multimodal, but this GGUF/runtime combination
was evaluated only for text/code generation. Vision behavior is not claimed.

## Reproducibility

Per-shard SHA-256 digests are provided in `SHA256SUMS`. The complete appendable
leaderboard, test protocol, harness hashes, and result-artifact hashes live in
Packwolf's `docs/benchmarks/MODEL_LEADERBOARD.md`.

## License and attribution

The upstream GLM-5.3-Flash checkpoint is MIT licensed. See `LICENSE` and the
upstream model card for attribution, limitations, and citation information.
