---
language:
- en
library_name: mlx
pipeline_tag: text-generation
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
- mlx
- mtplx
- apple-silicon
- mtp
- multi-token-prediction
- speculative-decoding
- moe
- conversational
- image-text-to-text
- ornith
---

# Ornith 1.5 35B-A3B — MTPLX / MLX

[Explore the model guide](https://huggingface.co/spaces/philipjohnbasile/local-ai-guide) · [All public work](https://huggingface.co/philipjohnbasile)

## Start here

| Question | Answer |
|---|---|
| What is it for? | Local experimentation with Ornith's 35B-A3B model through MTPLX on Apple Silicon. |
| What did I contribute? | Philip John Basile's MLX/MTPLX conversion and upload. The original model is by [Ornith AI](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B); conversion/runtime tooling is [MTPLX](https://github.com/youssofal/MTPLX). |
| What runs it? | MTPLX with the artifact's MTP contract. The Forge manifest records 2.9.0. Generic Transformers/vLLM/SGLang compatibility is not established. |
| What was validated? | A recorded Forge row per depth; CLI syntax was checked with a 2.8.0 (2.8.1) dry run. The card audit did not independently run loading, generation, vision, or a quality comparison. |
| How much memory? | About 22.09 GB of weight payload, including vision and MTP. Peak runtime memory and a minimum-memory configuration were not measured in the card audit. Allow additional space for the runtime, OS, and KV cache. |
| What are the limits? | Vision weights and 262K context are present/configured, not independently qualified here. The card documents conflicting license metadata; read the attribution and licensing section. |

### Quick start — documented MTPLX command shape

```bash
mtplx pull philipjohnbasile/ornith-ai-Ornith-1.5-35B-A3B-V2-MTPLX
mtplx start cli \
  --model philipjohnbasile/ornith-ai-Ornith-1.5-35B-A3B-V2-MTPLX \
  --profile sustained --depth 1
```

These commands resolve mutable repository state. The dry-run and runtime-version qualifications in **Usage** below apply; re-profile for your hardware and workload.

An Apple-Silicon MTPLX Forge artifact of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
with native multi-token prediction (MTP). The language-model body uses mixed
4-bit/8-bit affine MLX quantization, while the MTP sidecar is stored in BF16. The
vision weights and multimodal chat template are included.

This is an **MTPLX/MLX runtime artifact**, not a drop-in Transformers, vLLM,
or SGLang checkpoint. See the upstream model card for the model's training,
intended uses, and upstream evaluation results.

## Artifact summary

| Field | Value |
|---|---|
| Upstream model | [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| Pinned upstream revision | [`e4dfb35a93d4b6822a811a7676f3488514abe7e2`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/e4dfb35a93d4b6822a811a7676f3488514abe7e2) |
| Architecture | Qwen3.5 MoE conditional generation; MTPLX runtime arch `qwen3-next-mtp` |
| Parameters | Target index metadata reports 35,107,180,016 base-model parameters, including vision and excluding the MTP sidecar; the upstream card reports approximately 3B activated parameters per token |
| Body quantization | `config.json` specifies 4-bit affine quantization with group size 64 and 8-bit overrides for router/shared-expert gates |
| MTP | `mtp.safetensors` contains one MTP transformer layer and all 785 tensors are BF16; the manifest records a maximum speculative decode depth of D3 and selects D1 |
| Vision | Preserved in `model-vision.safetensors` (333 tensors) |
| Context configuration | 262,144 tokens in `config.json` |
| Weight payload | Approximately 22.09 GB: 19.51 GB language-model shards + 0.89 GB vision + 1.69 GB MTP |
| Manifest `mtplx_version` | 2.9.0 |
| Manifest recommended profile | `sustained` |

An earlier Hub snapshot reported `safetensors.total` as 5,865,901,936
stored BF16/U32 elements from the
indexed base-model and vision files; it excludes the separately named
`mtp.safetensors` sidecar. Because the quantized weights are packed, this
stored-element count is not the logical base-model parameter count; the target
index metadata reports 35,107,180,016 parameters, also excluding the MTP
sidecar. The rendered Hub badge showed **35B** when checked on September 10,
2026; the historical packed-element count should not be presented as the
current badge.

[`mtplx_runtime.json`](./mtplx_runtime.json) records selected conversion
settings, the claimed source repository and revision, an artifact-fingerprint
value, the MTP contract, and Forge verification rows. This audit did not
reproduce the conversion or independently recompute the fingerprint.
Quantization overrides and model structure are recorded in
[`config.json`](./config.json).

## Usage

The `mtplx start cli` argument form below was accepted by `--dry-run` using the
locally installed CLI, which reports `2.8.0 (2.8.1)`. The `mtplx pull` command
was not executed during this audit, although both command forms are present in
the tagged MTPLX 2.9.0 source. Model loading, generation, and visual-input
handling were not executed. The artifact manifest records MTPLX 2.9.0 as the
forge version; runtime compatibility with other releases is unverified. These
unpinned commands resolve mutable repository state.

```bash
mtplx pull philipjohnbasile/ornith-ai-Ornith-1.5-35B-A3B-V2-MTPLX

mtplx start cli \
  --model philipjohnbasile/ornith-ai-Ornith-1.5-35B-A3B-V2-MTPLX \
  --profile sustained \
  --depth 1
```

Hugging Face may also display generic MLX-LM examples for this repository. In
the inspected MLX-LM 0.31.3 standard loader, weight discovery uses
`model*.safetensors`, so the separately named `mtp.safetensors` is not loaded
and MTP is not activated. No MLX-LM load or generation run was performed. Use
MTPLX when MTP decoding is required.

The sampler used in the Forge verification record was:

```text
temperature=0.6, top_p=0.95, top_k=20
```

The included chat template is a reasoning/tool-use template. Applications
should handle thinking and tool-call output according to the MTPLX surface
they use.

## Forge verification

The Forge record selected **D1** for this artifact:

| Decode depth | Throughput | Relative to AR | Acceptance by proposed position |
|---:|---:|---:|---:|
| D0 (AR) | 116.84 tok/s | 1.000x | — |
| **D1** | **159.39 tok/s** | **1.364x** | **91.48%** |
| D2 | 143.00 tok/s | 1.224x | 92.21%, 21.10% |
| D3 | 120.93 tok/s | 1.035x | 91.47%, 18.69%, 0.91% |

- Publisher-supplied original card label: Apple M5 Max (not independently
  verified by this card repair)
- Machine-readable environment: `macOS-27.0-arm64-arm-64bit-Mach-O`; the JSON
  does not identify the chip model
- Timestamp: 2026-08-22 01:50:23 -04:00
- Artifact-recommended profile: `sustained`
- Sampler: temperature 0.6, top-p 0.95, top-k 20

These are one recorded, stop-terminated Forge verification row per depth on
this artifact and device. They are useful for choosing the local MTP depth;
they are **not** a statistically powered benchmark, a cross-model comparison,
or an independent quality evaluation. The record does not provide the prompt,
repeat count, output-token count, warmup procedure, RAM/core configuration, or
thermal/power state. Each recorded row sets `quality_passed: true`; because the
record omits the evaluated prompt/output, criterion, and exactness baseline,
this audit could not interpret or independently validate that flag.

## Provenance

`mtplx_runtime.json` records `forged_locally: true` from the pinned upstream
BF16 checkpoint and the following selected settings:

```text
body_bits=4
body_group_size=64
body_mode=affine
mtp_policy=keep_bf16
```

The runtime record contains this artifact-fingerprint value:

```text
sha256:f7d71fe2fb6c603c1a8a301556dc340b761a92cfd96d53c15312bce7842a4b77
```

Audited weight/config/runtime snapshot:
`5e00d3b9fb1ef9383b47f20b699975ad33a7cf1c`. The manifest's embedded
`published_to_hf.revision=c3ac80c3486344d0fabf9ca5771fbf04871efdf6` is
historical provenance, not the audited or current repository revision.

## Limitations and validation scope

- Only the MTPLX/MLX path and the recorded Forge verification are documented
  here. Direct use with Transformers, vLLM, SGLang, or other runtimes was not
  validated for this converted artifact.
- The `mtplx start cli` argument form was accepted by `--dry-run`; `mtplx pull`,
  model loading, and generation were not performed during this audit.
- The conversion has not been independently benchmarked against the upstream
  BF16 checkpoint for task quality, perplexity, long-context quality, or
  multimodal quality.
- The 262,144-token context length is configuration metadata, not a completed
  long-context acceptance test for this conversion.
- Vision weights are present, but the Forge speed record does not exercise a
  visual-input workload.
- MTP acceptance and throughput depend on prompt, output length, sampling,
  runtime version, thermal state, and hardware. Re-profile for your workload.
- This conversion inherits the upstream model's behavioral limitations and
  may produce inaccurate, biased, unsafe, or otherwise undesirable output.

## License and attribution

At the audited revisions, the [pinned source's Hugging Face metadata](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/e4dfb35a93d4b6822a811a7676f3488514abe7e2/README.md)
declares `license: mit`, but neither the source nor this conversion contains a
standalone `LICENSE` file, and the source's `license_link` resolves to a missing
file. `mtplx_runtime.json` separately records
`published_to_hf.license_spdx: apache-2.0`; this audit treats that as
conflicting manifest metadata, not a license grant. Confirm all applicable
upstream terms before use or redistribution.

Upstream model and research: [Ornith AI](https://huggingface.co/ornith-ai)

Conversion/runtime tooling: [MTPLX](https://github.com/youssofal/MTPLX)

Conversion and upload: [Philip John Basile](https://huggingface.co/philipjohnbasile)
