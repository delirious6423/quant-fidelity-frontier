---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- mtplx
- mxfp4
- native-mtp
- abliterated
- vision-language
- qwen3.8
---

# Qwen3.8-27B Abliterated MLX MXFP4 + Native MTP

Vision-enabled MTPLX package of `Qwen/Qwen3.8-27B`, pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with a measured refusal-direction edit.

- Language model: MXFP4, group size 32; strength 2.5 norm-preserving edit on 128 modules
- Vision tower: unchanged same-revision BF16 weights
- Speculation: 15 unchanged native BF16 MTP tensors; recommended depth 3
- Runtime: MTPLX 2.0.2 or newer

The frozen held-out behavior suite changed measured refusal markers from 12/12 to 0/12,
with benign-sensitive refusal at 0/12 and utility at 6/6. MTPLX tensor inspection, text
generation, and three vision requests through its API passed. This is scoped evidence, not a
claim that the model is universally “uncensored.”

The Hub's approximately 5.5B safetensors count reflects packed MXFP storage elements; the
underlying architecture remains the full 27B model.

```bash
mtplx quickstart \
  --model Shiftedx/Qwen3.8-27B-Abliterated-MLX-MXFP4-MTP \
  --mtp --depth 3 --profile sustained
```

This edit reduces refusal behavior and may increase harmful or incorrect outputs. Treat prompts,
images, and outputs as untrusted; do not submit secrets, and sandbox tools or generated code with
least-privilege access. This package adds no telemetry or remote execution.

The upstream Apache-2.0 license and model limitations continue to apply.

<!-- shiftedx-bench:start -->
## Shiftedx Bench post-publication qualification

This table was generated from the frozen lightweight quant gate after the model weights were published. Categories remain separate; the benchmark does not produce a composite intelligence score.

| Lane | Passed | Accuracy | Mean wall time | Mean decode | Peak active memory |
|---|---:|---:|---:|---:|---:|
| Quality | 6/10 | 60.0% | 22.41 s | 53.46 tok/s | 32.36 GiB |
| Long context | 13/15 | 86.7% | 145.65 s | 42.63 tok/s | 41.02 GiB |
| Tool calling | 6/6 | 100.0% | 4.26 s | 41.08 tok/s | 35.18 GiB |
| Agentic | 0/2 | 0.0% | 19.22 s | — tok/s | — |
| Vision | 3/4 | 75.0% | 4.81 s | 49.61 tok/s | 34.42 GiB |

- Tested model revision: [`d6352f90f36268551b9032b37ec92fe89c1a4ebe`](https://huggingface.co/Shiftedx/Qwen3.8-27B-Abliterated-MLX-MXFP4-MTP/tree/d6352f90f36268551b9032b37ec92fe89c1a4ebe)
- Benchmark: [Shiftedx Bench v0.3.0](https://huggingface.co/datasets/Shiftedx/shiftedx-bench/tree/3bbb0bfa01e33503163cb34ef52b4d507e456265)
- Context lengths represented: 4,096, 16,384, 65,536, 131,072 prompt tokens; effective tested context: 131,072 tokens
- Runtime contract: MTPLX 2.7.1 sustained; thinking on/medium; sampler temperature=1.0, top_p=0.95, top_k=20; Hermes hybrid routing with native explicit-parallel calls; KV cache `off`; MTP depth `3`
- Host: Apple M4 Max, 64 GiB unified memory
- Total measured request wall time: 2492.15 seconds
- 260,096-token status: not run; it is outside the lightweight quant gate.

Scores are specific to the linked model revision, benchmark revision, runtime contract, and host. Changing weight precision, KV-cache precision, reasoning mode, template, or speculative depth creates a different benchmark candidate.
<!-- shiftedx-bench:end -->
