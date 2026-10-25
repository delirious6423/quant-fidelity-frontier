---
base_model:
- google/gemma-4-26B-A4B-it
---

[Upstream model](https://huggingface.co/google/gemma-4-26B-A4B-it)

This model has been quantized with the following recipe, intended for high-throughput batch generation:

- All expert projections are NVFP4.
- All attention projections (Q, K, V, and O) are MXFP8. The per-layer shared MLP is also MXFP8.
- KV cache is FP8 with static per-tensor scale factors.
- The vision encoder, embeddings, expert router, and other auxiliary parameters remain in their original precision (BF16 or FP32).

# Evaluation

The quantized model has been verified to preserve its instruction-following capabilities:

### Thinking on

| Benchmark | Metric                    | BF16  | Quantized | Recovery |
|-----------|---------------------------|-------|-----------|----------|
| IFEval    | prompt-level strict       | 94.41 | 94.69     | 100.30%  |
|           | instruction-level strict  | 96.25 | 96.40     | 100.16%  |
| IFBench   | prompt-level strict       | 76.92 | 76.58     | 99.56%   |
|           | instruction-level strict  | 79.07 | 78.92     | 99.81%   |

### Thinking off

| Benchmark | Metric                    | BF16  | Quantized | Recovery |
|-----------|---------------------------|-------|-----------|----------|
| IFEval    | prompt-level strict       | 90.11 | 89.46     | 99.28%   |
|           | instruction-level strict  | 93.35 | 92.75     | 99.36%   |
| IFBench   | prompt-level strict       | 43.00 | 42.33     | 98.44%   |
|           | instruction-level strict  | 47.38 | 47.17     | 99.56%   |

No other benchmarks have been run.
