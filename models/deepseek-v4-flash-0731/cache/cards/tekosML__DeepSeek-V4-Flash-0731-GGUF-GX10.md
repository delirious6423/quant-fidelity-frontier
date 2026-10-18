---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: gguf
tags:
  - gguf
  - deepseek
  - deepseek-v4
  - mixture-of-experts
  - dgx-spark
  - gx10
  - gb10
  - ds4
---

# DeepSeek V4 Flash 0731 GGUF for one GX10

This is a community GGUF quantization of [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), built to run fully resident on one 128 GB NVIDIA GB10 system such as an ASUS Ascent GX10 or DGX Spark.

It is not an official DeepSeek release.

## Artifact

| Property | Value |
|---|---|
| File | `DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf` |
| Size | 86,720,111,552 bytes, 80.76 GiB |
| SHA-256 | `73b86c93e7b0be37f68c4b3a5361c7e17f9b278647aa335ea304ac5dfd784867` |
| Source revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Tested runtime | [`antirez/ds4`](https://github.com/antirez/ds4) at `54b36ed9ba42da31b24f2d1a5feb075c2475dbb1` |

The GGUF contains the 45-shard base target model. It does not include shards 46 to 48 or enable the optional DSpark speculative module.

## Quantization layout

The layout follows the proven one-GB10 V4 Flash recipe:

- routed gate and up experts: `IQ2_XXS`
- routed down experts: `Q2_K`
- attention projections: `Q8_0`
- shared experts: `Q8_0`
- output head: `Q8_0`
- token embedding: `F16`
- sensitive norms and control tensors: `F16` or `F32` according to the pinned template

The calibration imatrix is Antirez's 1.5M-token routed-MoE dataset artifact from `antirez/deepseek-v4-gguf`, revision `a88c423b511666d7ff7a4dcaee651669312bea97`.

### Where the 80.76 GiB goes

A header-only GGUF audit reconciled every serialized tensor byte without reading tensor payloads:

| Tensor family or type | Stored payload |
|---|---:|
| Routed experts | 72.562 GiB, 89.845% of the file |
| `IQ2_XXS` tensors | 44.344 GiB |
| `Q2_K` tensors | 28.219 GiB |
| `Q8_0` tensors | 6.146 GiB |
| `F16` tensors | 2.041 GiB |
| Attention projections | 5.458 GiB |
| Shared experts | 1.071 GiB |

The model routes 6 of 256 experts per layer. This does not imply that only 6/256 of stored expert bytes are transferred by the runtime, but it explains why selective low-bit expert storage matters more than uniformly reducing every tensor. See `measurement/header-byte-accounting.json`.

Important limitation: the imatrix was collected for the earlier V4 Flash checkpoint and reused for 0731. This release has not established broad quality equivalence to the official mixed-precision checkpoint.

## Measured GX10 result

Measured on one NVIDIA GB10 with 128 GB unified memory using target-only decoding. Both checkpoints used the same DS4 binary, prompts, tokenizer/template path, context, sampler settings, seed, power setting, and GPU budget.

Protocol:

- 1,024-token context
- 256-token output cap
- greedy decoding, temperature 0
- seed `260729`
- non-thinking mode
- one warm-up per arm and workload
- three measured runs per arm and workload
- balanced `ABBAAB` measured order

| Workload | Earlier V4 Flash | V4 Flash 0731 | Change |
|---|---:|---:|---:|
| Code generation | 16.86 tok/s | **16.91 tok/s** | +0.30% |
| Technical prose | 16.83 tok/s | **16.84 tok/s** | +0.06% |

Across all 16 warm-up and measured runs:

- all runs completed
- zero safety breaches
- zero swap growth
- minimum available memory was approximately 14.0 GiB
- maximum sampled temperature was 67 C
- maximum sampled power was 41.69 W
- all three measured outputs were byte-identical within each checkpoint and workload

The output text differed between the two checkpoints, as expected after post-training. The benchmark prompts were capped at 256 generated tokens and therefore do not support a broad quality-retention claim. See `benchmark/protocol.json` and `benchmark/summary.json`.

The matched benchmark ran before a same-length provenance-only metadata sanitization. That edit replaced the local `quantize.imatrix.file` string without changing tensor payload bytes, tensor offsets, or file size. The uploaded artifact was rehashed, header-audited, and smoke-tested afterward.

## Verified smoke test

The final GGUF loaded through the CUDA backend on an NVIDIA GB10 and generated:

> This sanitized local model response was generated on an NVIDIA GX10.

For that post-sanitization one-shot smoke test:

- CUDA model preparation: 19.933 seconds
- resident model spans: 80.76 GiB
- planned model plus 1,024-token KV memory: 80.86 GiB
- prefill: 40.52 tok/s
- generation: 15.56 tok/s

The matched three-run benchmark above is the better throughput reference.

## Download

```bash
hf download tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10 \
  DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf \
  --local-dir .
```

## Run with DS4

Build DS4 at the tested commit using its documented CUDA Spark target, then run:

```bash
./ds4 \
  -m DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf \
  --prompt-file prompt.txt \
  --ctx 1024 \
  -n 256 \
  --temp 0 \
  --seed 260729 \
  --nothink \
  --power 100 \
  --gpu-vram 105
```

Start conservatively. Larger context windows require more KV memory and were not validated for this release.

## Reproducibility and lineage

See:

- `provenance.json` for revisions, sizes, and hashes
- `READY.json` for the exercised smoke-test result
- `benchmark/protocol.json` and `benchmark/summary.json` for the matched GX10 benchmark
- `measurement/header-byte-accounting.json` for exact GGUF payload and tensor-family accounting
- `conversion/resume-notes.md` for the boundary-validated write-resume provenance after the original converter stopped at tensor 133

The resume helper changed file continuation behavior only. It validated the complete GGUF header, every planned tensor's name, shape, type, size, and offset, and required the existing file to end at an exact tensor boundary before appending the remaining tensors. The final artifact was independently hashed after completion.

## License

The source checkpoint is MIT licensed. The original license is included in this repository. Preserve the license and attribution when redistributing this derivative.
