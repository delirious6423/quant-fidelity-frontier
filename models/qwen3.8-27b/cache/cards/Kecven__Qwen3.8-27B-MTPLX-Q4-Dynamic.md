---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - mlx
  - mtplx
  - qwen
  - qwen3.8
  - mtp
  - speculative-decoding
  - quantized
  - dynamic-quantization
  - mixed-precision
library_name: mlx
pipeline_tag: text-generation
---

# Qwen3.8-27B-MTPLX-Q4-Dynamic

Dynamic mixed-precision MTPLX conversion of `Qwen/Qwen3.8-27B`.

The model uses Q4 for most tensors while preserving higher precision for selected sensitive components. Native MTP weights are kept in BF16.

## Quantization

Tensor distribution:

| Precision | Tensor count |
|---|---:|
| Q4 | 602 |
| Q6 | 392 |
| Q8 | 4 |

The dynamic quantization strategy uses:

- Q4 for the majority of model weights
- Q6 for selected sensitive projections and transformer layers
- Q8 for selected high-sensitivity input/output components
- BF16 for native MTP weights

Local model size is approximately **18 GB** before accounting for filesystem reporting differences.

## MTP

- Native MTP weights: BF16
- Maximum verified MTP depth: 3
- Forge verdict: `mtp_depth_wins`
- All tested depths passed the Forge quality gate

## Verification Results

Verified locally with MTPLX Forge.

| Mode | Speed | vs AR | Acceptance |
|---|---:|---:|---|
| AR / depth 0 | 14.12 tok/s | 1.00× | — |
| MTP depth 1 | 25.81 tok/s | 1.83× | 97.76% |
| MTP depth 2 | 31.90 tok/s | 2.26× | 96.85% / 91.34% |
| MTP depth 3 | **40.57 tok/s** | **2.87×** | 96.81% / 90.04% / 83.67% |

Best verified configuration:

```text
AR:           14.12 tok/s
MTP depth 3: 40.57 tok/s
Speedup:      2.87×
```

## Comparison with Other Builds

All variants below were verified on the same local system.

| Build | Approx. size | AR | MTP D3 | MTP speedup |
|---|---:|---:|---:|---:|
| Q4 | ~15–16 GB | 15.74 tok/s | **44.34 tok/s** | 2.82× |
| Mixed 4/6 | ~16 GB | 16.11 tok/s | 42.14 tok/s | 2.62× |
| **Q4 Dynamic** | **~18 GB** | 14.12 tok/s | **40.57 tok/s** | **2.87×** |
| Q8 | ~28–30 GB | 9.36 tok/s | 29.03 tok/s | 3.10× |

The flat Q4 build provides the highest absolute throughput.

This Dynamic variant trades approximately 8–9% of MTP D3 throughput relative to flat Q4 for substantially higher precision across hundreds of selected tensors.

## MTP Acceptance

At depth 3:

```text
Position 1: 96.81%
Position 2: 90.04%
Position 3: 83.67%
```

The third speculative token is accepted more than 83% of the time in this verification run.

## Quantization Strategy

The model was first converted with a custom MLX-LM quantization predicate.

Broadly:

```text
Most tensors                  → Q4
Sensitive projections/layers → Q6
Selected input/output tensors → Q8
Native MTP sidecar            → BF16
```

The already-quantized MLX trunk was then passed to MTPLX Forge as an `mlx_affine_with_mtp` source.

Forge therefore preserved the existing mixed-precision trunk instead of requantizing it as flat Q4.

## Related Builds

Higher throughput:

`Kecven/Qwen3.8-27B-MTPLX-Q4`

Higher weight precision:

`Kecven/Qwen3.8-27B-MTPLX-Q8`

## Performance Notes

Absolute throughput depends heavily on Apple Silicon generation, memory bandwidth, context length, thermals, and runtime configuration.

The relative AR-to-MTP measurements are generally more useful than comparing absolute token rates across different machines.

## License

Apache-2.0, following the license of the original `Qwen/Qwen3.8-27B` model.
