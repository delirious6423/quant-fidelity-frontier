---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3.8-Flash-Next
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
- qwen4-exp
- MXFP4
- MXFP4
- mtp
- vision
---

# AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP — 5.73 BPW measured main

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. The language path is quantized while the multi-token-prediction (MTP) head and vision tower are preserved at BF16 in the checkpoint (or a bound sidecar when present).

> **Development evidence — not a certified AXQuant release.** This package has conversion and
> artifact-integrity records, but it does not publish measured quality, long-context, kernel-speed,
> or MTP-speed evidence. Do not interpret the AXQ product label as a benchmark claim.


## Model details

| Property | Value |
| --- | --- |
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) |
| Source revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Product family | `qwen4-exp` |
| Source architecture | `Qwen4ExpForConditionalGeneration` (mixture of experts (MoE)); text path optimized |
| Main-model parameters | 177.39B logical parameters |
| Quantizer | AXQuant `1.9.0` |
| Hub budget class | `MXFP4` |
| AXQuant base precision class | `6p8bpw` |
| Planned storage-adjusted BPW | 6.3924 |
| Measured main-model BPW | 5.7281 |
| Measured total BPW, including MTP | **5.8769** |
| Safetensors weight size | 132.23 GB |
| Approximate complete download | 132.26 GB |
| Configured maximum context | 262,144 tokens; practical limits depend on unified memory |
| Primary MLX runtime | MLX-LM |
| AX Engine native execution | Not established; no validated native manifest is included |
| MTP present | `True` |
| Vision present | `True` |
| Audio present | `False` |

This repository contains MLX Safetensors. It does **not** contain PyTorch or GGUF weights.

## Choosing an AXQ pack

AXQ names describe a **storage-budget product class**, not one uniform precision applied to every
tensor. Protected tensors remain at higher precision, so the exact measured BPW is authoritative.
In particular, a `6bit`-named mixed plan may retain `4bit` as its base precision while selecting
6-bit, 8-bit, or BF16 for other tensors to meet an approximately 6-BPW total budget. Protection
floors can also raise a `4bit`-named pack close to (or above) a `6bit` budget on small or heavily
protected models. When that collapse happens, AutomatosX does **not** publish a separate
misleading `4bit` sibling for that base.


| Sibling | Intended trade-off |
| --- | --- |
| [4bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP) | Lower-storage AXQ budget; check its exact BPW |
| [6bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP) | Higher average precision near the 6-BPW budget |

See the [AutomatosX collections](https://huggingface.co/AutomatosX/collections)
for the family catalog, or the [complete index](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

## Download

```bash
python -m pip install -U huggingface_hub
hf download AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP --local-dir ./AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP
```

Allow at least 132.26 GB of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-LM

```bash
python -m pip install -U mlx-lm
mlx_lm.generate \
  --model AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP \
  --prompt "Explain mixed-precision quantization in three sentences." \
  --max-tokens 128 \
  --temp 0.0
```

MLX-LM compatibility covers standard **text/backbone inference**. It may ignore AXQuant runtime
metadata and optional sidecars (`vision.safetensors`, `mtp.safetensors`); this command therefore
does not establish MTP acceleration or vision-language quality. The artifact records MLX
`0.32.1` and MLX-LM `0.31.3` from conversion.

## AX Engine status

This package does **not** include a validated native `model-manifest.json`, so AX Engine execution
is not established by this release. The AX Engine fields in `axquant_runtime.json` describe the
intended compatibility contract, not observed runtime evidence. Use the architecture-specific MLX
runtime path above. The artifact records AX Engine version
`7.5.7`, but version discovery alone is not a runtime check.

## Use the packaged Qwen MTP head with oMLX or MTPLX

Use the explicit runtime branch rather than importing this original layout:

| Runtime | Verified variant | Download revision |
| --- | --- | --- |
| oMLX 0.7.0.dev4 | [oMLX branch](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP/tree/ec729dc15eb3c2559939acd3e89f76eb56f3aef6) | `omlx` |
| MTPLX 2.11.3 | [MTPLX branch](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP/tree/3793ab3412d45fbe98a031bd9b9244ad59420f17) | `mtplx` |

```bash
hf download AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP --revision omlx --local-dir ./flash-next-omlx
hf download AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP --revision mtplx --local-dir ./flash-next-mtplx
```

Download each branch into a fresh directory and follow its model card and
manifest launch requirements. The oMLX branch indexes native Qwen4 MTP and
requires per-model `qwen4_ple_ssd_offload=true` on the tested host. The MTPLX
branch supplies a canonical n-gram table plus adapted norms/native MTP experts;
image input requires MTP mode. Original MXFP4 precision is retained.

Both variants passed short text generation with MTP off/on, plus image
generation in supported modes, on 2026-10-06. Branch smoke summaries bind the
exact tested artifacts. These are development compatibility checks, not
quality, MTP exactness, throughput, or AXQuant certification. The historical
`qwen3-next-mtp` label in this original main branch does not establish native
Flash-Next compatibility. Original checkpoint files and historical evidence
remain on main; peer variants carry their own manifests.

## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
| `4bit` | 123.73B | 68.74% |
| `8bit` | 52.57B | 29.20% |
| `bf16` | 3.70B | 2.05% |

- Quantization methods: `affine, bf16`.
- Group sizes used by quantized assignments: `32, 64`.
- MTP sidecar: 31 tensors, 2.61B parameters, 5.21 GB, BF16.
- Vision sidecar: not included.
- Vision weights: protected BF16 in main shards.
- Optimization scope: `text-path`.
- Support tier: `convertible`.

BF16 sidecars, when present, are included in total download size. Their presence does not by itself
establish MTP acceleration or vision-language quality.

## Evidence and validation status

| Check | Status |
| --- | --- |
| Planning evidence | `architecture_prior` |
| Calibration | none; the allocation is based on architecture priors |
| Quantizer execution | 708/708 recorded module conversions succeeded; 0 fallbacks |
| AX Engine native manifest | not included |
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
- AX Engine execution is not established because this package has no validated native manifest.

- Upstream capabilities, limitations, biases, and responsible-use guidance still apply.

## Provenance and audit files

- [`axquant_manifest.json`](axquant_manifest.json): package identity, byte accounting, runtime
  contract, software versions, and file checksums.
- [`axquant_plan.json`](axquant_plan.json): per-tensor precision decisions and planning evidence.
- [`axquant_quantizer_execution.json`](axquant_quantizer_execution.json): conversion coverage and
  fallback records.
- [`axquant_runtime.json`](axquant_runtime.json): declared AX Engine and MLX compatibility metadata; runtime checks remain separate evidence.
- [`axquant_mtp_sidecar_manifest.json`](axquant_mtp_sidecar_manifest.json): MTP tensor provenance.

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
[`LICENSE`](LICENSE) and the [Qwen/Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) for license terms, model
limitations, and responsible-use guidance.

## Catalog runtime audit (2026-10-06)

See [runtime_audit.json](runtime_audit.json) for pinned source config/index/header
bindings and immutable measured runtime-variant references. Main retains the
original checkpoint layout; download the named peer branch for its runtime.
