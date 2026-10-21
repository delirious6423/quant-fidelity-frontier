---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - mlx
  - mtplx
  - qwen
  - qwen3.8
  - speculative-decoding
  - mtp
  - quantized
library_name: mlx
pipeline_tag: text-generation
---

# Qwen3.8-27B-MTPLX-Q8

MTPLX-compatible 8-bit conversion of `Qwen/Qwen3.8-27B`, built from the original BF16 checkpoint while preserving the model's native MTP weights in BF16.

## Quantization

- Base model: `Qwen/Qwen3.8-27B`
- Body: 8-bit affine
- Group size: 64
- MTP sidecar: native BF16
- Maximum verified MTP depth: 3
- Source format: native BF16 + MTP

Forge recipe:

```json
{
  "body_bits": 8,
  "body_dtype": "auto",
  "body_group_size": 64,
  "body_mode": "affine",
  "mtp_policy": "keep_bf16"
}
```

## Verification Results

The model was verified locally with MTPLX Forge.

| Mode | Speed | vs AR | Acceptance |
|---|---:|---:|---|
| AR / depth 0 | 9.36 tok/s | 1.00× | — |
| MTP depth 1 | 18.84 tok/s | 2.01× | 100.00% |
| MTP depth 2 | 26.84 tok/s | 2.87× | 95.92% / 93.20% |
| MTP depth 3 | **29.03 tok/s** | **3.10×** | 97.38% / 90.09% / 83.67% |

Forge verdict:

```text
mtp_depth_wins
```

All tested MTP depths passed the Forge quality verification gate.

## Performance

The strongest verified configuration was MTP depth 3:

```text
AR:           9.36 tok/s
MTP depth 3: 29.03 tok/s
Speedup:      3.10×
```

The third speculative position was still accepted approximately 83.7% of the time.

These throughput numbers are hardware-specific and should primarily be used as relative measurements between AR and MTP on the same machine.

## Q4 vs Q8

A separate 4-bit build of the same model was also tested on the same system.

| Quantization | AR | MTP D3 | MTP Speedup |
|---|---:|---:|---:|
| Q4 | 15.74 tok/s | **44.34 tok/s** | 2.82× |
| Q8 | 9.36 tok/s | **29.03 tok/s** | 3.10× |

The Q8 build achieves a larger relative MTP multiplier, but the Q4 build has substantially higher absolute throughput because the smaller model body requires less memory bandwidth.

The Q8 variant is intended for users who prefer higher weight precision and can accept the additional memory usage and lower decode throughput.

## MTP Acceptance

At the selected depth 3:

```text
Position 1: 97.38%
Position 2: 90.09%
Position 3: 83.67%
```

This high acceptance rate allows native MTP speculative decoding to provide a substantial speedup over standard autoregressive decoding.

## Architecture

```text
Base architecture: Qwen3.8-27B
MTPLX architecture: qwen3-next-mtp
Body precision: 8-bit affine
MTP precision: BF16
Maximum MTP depth: 3
```

The native MTP weights were preserved from the original checkpoint and were not requantized.

## Usage

This model is intended for MTPLX on Apple Silicon.

MTPLX can use the included `mtplx_runtime.json` metadata to determine the MTP contract, supported speculative depth, sampler configuration, and locally verified performance profile.

## Related Model

For significantly higher throughput with a smaller memory footprint, see:

`Kecven/Qwen3.8-27B-MTPLX-Q4`

The Q4 build reached approximately **44.34 tok/s** at MTP depth 3 on the same verification setup.

## License

Apache-2.0, following the license of the original `Qwen/Qwen3.8-27B` model.