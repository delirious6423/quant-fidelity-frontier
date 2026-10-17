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
---

# Qwen3.8-27B-MTPLX-Q4

MTPLX-compatible 4-bit conversion of [Qwen/Qwen3.8-27B], built locally from the original BF16 checkpoint while preserving the model's native MTP weights in BF16.

## Quantization

- Base model: `Qwen/Qwen3.8-27B`
- Body: 4-bit affine
- Group size: 64
- MTP sidecar: native BF16
- Maximum verified MTP depth: 3
- MTPLX version: 2.3.0
- Source format: BF16 native MTP

Forge recipe:

```json
{
  "body_bits": 4,
  "body_dtype": "auto",
  "body_group_size": 64,
  "body_mode": "affine",
  "mtp_policy": "keep_bf16"
}
```

## Verification Results

Verified locally with MTPLX Forge using the `long-code-uncapped` suite with a maximum generation budget of 2048 tokens.

| Mode | Speed | vs AR | Acceptance |
|---|---:|---:|---|
| AR / depth 0 | 15.74 tok/s | 1.00× | — |
| MTP depth 1 | 27.82 tok/s | 1.77× | 96.95% |
| MTP depth 2 | 36.53 tok/s | 2.32× | 96.13% / 90.66% |
| MTP depth 3 | **44.34 tok/s** | **2.82×** | 95.90% / 89.55% / 81.72% |

Forge verdict:

```text
mtp_depth_wins
```

All tested depths passed the Forge quality verification gate.

The recommended profile produced by Forge is:

```text
sustained
```

## MTP Configuration

```text
architecture: qwen3-next-mtp
mtp_depth_max: 3
mtp_sidecar: bf16
base_hidden_variant: post_norm
hidden_variant: post_norm
concat_order: embedding_hidden
mtp_position_mode: local
```

## Source Provenance

- Source repository: `Qwen/Qwen3.8-27B`
- Source commit: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Conversion: MTPLX Forge
- MTPLX version: `2.3.0`
- Forged locally on macOS / Apple Silicon

The MTP weights were preserved from the original BF16 checkpoint rather than requantized.

## Performance Notes

The benchmark numbers above are specific to the machine used for verification and should not be treated as universal throughput figures.

The more useful result is the relative improvement on the same hardware:

```text
15.74 tok/s AR
        ↓
44.34 tok/s MTP depth 3

2.82× decode throughput
```

MTP acceptance remained high through depth 3, with the third speculative position accepted approximately 81.7% of the time.

## Usage

This model is intended for use with MTPLX on Apple Silicon.

After downloading, inspect the included runtime metadata and use the MTPLX runtime to load the model. The Forge-generated runtime metadata contains the verified MTP contract and recommended configuration.

## License

This conversion follows the Apache-2.0 license of the original `Qwen/Qwen3.8-27B` checkpoint.

See the upstream model repository for additional usage information and attribution requirements.