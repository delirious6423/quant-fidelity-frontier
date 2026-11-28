---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- quantized
- mixed-precision
- axquant
- axq
- development
- qwen3.8
- MXFP4
- mtp
- vision
---

# AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP — 4.82 BPW measured main

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. The language path is quantized while the multi-token-prediction (MTP) head and vision tower are preserved at BF16 in the checkpoint (or a bound sidecar when present).

> **Development evidence — not a certified AXQuant release.** This package has conversion and
> artifact-integrity records, but it does not publish measured quality, long-context, kernel-speed,
> or MTP-speed evidence. Do not interpret the AXQ product label as a benchmark claim.


## Model details

| Property | Value |
| --- | --- |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Product family | `qwen3.8` |
| Source architecture | `Qwen3_5ForConditionalGeneration` (dense); text path optimized |
| Main-model parameters | 27.36B logical parameters |
| Quantizer | AXQuant `3.0.1` |
| Hub budget class | `MXFP4` |
| Storage formats | `MXFP4, MXFP8, BF16` |
| Planned storage-adjusted BPW | 4.9917 |
| Measured main-model BPW | 4.8209 |
| Measured total BPW, including MTP | **4.9918** |
| Safetensors weight size | 17.33 GB |
| Approximate complete download | 17.37 GB |
| Configured maximum context | 262,144 tokens; practical limits depend on unified memory |
| Primary MLX runtime | MLX-LM |
| AX Engine native execution | Native manifest included; execution still requires a runtime check |
| DFlash2 | Declared target for `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde`; AX Engine selects the drafter. Metadata, not route or speed evidence |
| MTP present | `True` |
| Vision present | `True` |
| Audio present | `False` |

This repository contains MLX Safetensors. It does **not** contain PyTorch or GGUF weights.

## Choosing an AXQ pack

This family publishes MXFP4, MXFP8, and one CUDA NVFP4 W4A16 pack. The two MLX packs use
the same role policy at different block formats. Protected tensors stay at higher precision,
so measured BPW is the figure to compare. The CUDA pack is quantized from the same BF16 source and is not a requantization of the MLX weights. Activations stay BF16.


| Sibling | Intended trade-off |
| --- | --- |
| [MXFP8 sibling](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP8-MTP) | Same AXQ role policy at the other MLX block format. Vision, MTP, and norms stay BF16. |
| [CUDA NVFP4 W4A16](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-CUDA-AXQ-NVFP4-W4A16-MTP) | Native NVIDIA checkpoint quantized from the same BF16 source. Activations stay BF16. |

See the [AutomatosX collections](https://huggingface.co/AutomatosX/collections)
for the family catalog, or the [complete index](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

## Download

```bash
python -m pip install -U huggingface_hub
hf download AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP --local-dir ./AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP
```

Allow at least 17.37 GB of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-LM

```bash
python -m pip install -U mlx-lm
mlx_lm.generate \
  --model AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP \
  --prompt "Explain mixed-precision quantization in three sentences." \
  --max-tokens 128 \
  --temp 0.0
```

MLX-LM compatibility covers standard **text/backbone inference**. It may ignore AXQuant runtime
metadata and optional sidecars (`vision.safetensors`, `mtp.safetensors`); this command therefore
does not establish MTP acceleration or vision-language quality. The artifact records MLX
`0.32.4` and MLX-LM `0.32.2` from conversion.

## Serve with AX Engine and MTP

After installing AX Engine, download the complete repository (see
[AXQuant](https://github.com/defai-digital/axquant) for conversion, certificates, and
model-card tooling) and serve the local directory:

```bash
ax-engine serve ./AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP --port 31418
```

AX Engine is the authority for the AXQ runtime contract and native MTP sidecar.
This development package does not claim runtime speedups until identical-checkpoint benchmarks are
published. The artifact records AX Engine version `8.1.0`. Native
`model-manifest.json` status: included as `model-manifest.json`.

## Use the packaged Qwen MTP head with oMLX or MTPLX

This repository keeps the Qwen MTP head in `mtp.safetensors`; stock MLX-LM does not load that
sidecar by itself. Download the complete repository to a writable local directory before using a
sidecar-aware runtime.

In oMLX `0.6.3rc2` or newer, add the local directory, open **Model Settings**, choose
**Import MTP side-car**, and then enable **Lightning MTP**. The import changes only the local copy
so the MTP tensors become visible through the checkpoint index.

MTPLX can consume the packaged sidecar directly:

```bash
mtplx quickstart \
  --model ./AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP \
  --profile stable \
  --depth 1 \
  --reasoning off
```

`mtplx_runtime.json` declares the canonical `qwen3-next-mtp` execution contract. This enables
strict runtime discovery; it does not extend AXQuant quality, exactness, or speed certification to
oMLX or MTPLX.

## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
| `MXFP4` | 24.35B | 87.65% |
| `MXFP8` | 2.54B | 9.15% |
| `BF16` | 888.07M | 3.20% |

- Storage formats: `MXFP4, MXFP8, BF16`; MXFP4 and MXFP8 use 32-weight blocks with one shared E8M0 scale.
- Group sizes used by quantized assignments: `32`.
- MTP sidecar: 15 tensors, 424.70M parameters, 0.85 GB, BF16.
- Vision sidecar: 333 tensors, 460.73M parameters, 0.92 GB, BF16.
- Vision weights: protected BF16 sidecar.
- Optimization scope: `text-path`.
- Support tier: `convertible`.

BF16 sidecars, when present, are included in total download size. Their presence does not by itself
establish MTP acceleration or vision-language quality.

## Evidence and validation status

| Check | Status |
| --- | --- |
| Planning evidence | `architecture_prior` |
| Calibration | none; the allocation is based on architecture priors |
| Quantizer execution | 498/498 recorded module conversions succeeded; 0 fallbacks |
| AX Engine native manifest | included as `model-manifest.json` |
| Quality versus BF16 or uniform baselines | Not published; no quality-retention claim |
| MTP acceptance and speed | not measured; no MTP speedup claim |
| AX Engine kernel evidence | `unmeasured` |
| Vision-language quality | Not evaluated or claimed; vision tensors are preserved at BF16 |
| Speech-recognition quality | Not applicable |
| Long-context quality | 262,144-token capacity is config metadata, not a validated claim |
| Release certification | **Not certified**; formal AXQuant M0-M8 gates are not closed |

## Intended use and limitations

- Intended for local development and evaluation on Apple Silicon with MLX-compatible runtimes.
- No minimum unified-memory figure is claimed; loadability depends on model size, context length,
  KV-cache policy, runtime buffers, and other processes using unified memory.
- Architecture-prior allocation is not measured sensitivity. It must not be presented as measured
  model quality.
- MTP requires a sidecar-aware runtime. oMLX/MTPLX discovery compatibility does not establish exactness or speed certification for those runtimes.
- Vision weights are preserved at BF16, but this release does not claim validated VLM quality.
- The configured context window can require substantially more memory as the KV cache grows.

- Upstream capabilities, limitations, biases, and responsible-use guidance still apply.

## Provenance and audit files

- [`axquant_manifest.json`](axquant_manifest.json): package identity, byte accounting, runtime
  contract, software versions, and file checksums.
- [`axquant_plan.json`](axquant_plan.json): per-tensor precision decisions and planning evidence.
- [`axquant_quantizer_execution.json`](axquant_quantizer_execution.json): conversion coverage and
  fallback records.
- [`axquant_runtime.json`](axquant_runtime.json): declared AX Engine and MLX compatibility metadata; runtime checks remain separate evidence.
- [`axquant_mtp_sidecar_manifest.json`](axquant_mtp_sidecar_manifest.json): MTP tensor provenance.
- [`axquant_vision_sidecar_manifest.json`](axquant_vision_sidecar_manifest.json): protected vision tensor provenance.
- [`model-manifest.json`](model-manifest.json): AX Engine native tensor manifest.

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
[`LICENSE`](LICENSE) and the [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) for license terms, model
limitations, and responsible-use guidance.
