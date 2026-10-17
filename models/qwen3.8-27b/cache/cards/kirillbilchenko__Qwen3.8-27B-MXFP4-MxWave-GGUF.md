---
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: kirillbilchenko
inference: false
tags:
  - qwen
  - qwen3.8
  - qwen3_5
  - gguf
  - mxfp4
  - ollama
  - llama.cpp
  - multimodal
  - mtp
  - speculative-decoding
  - dgx-spark
  - blackwell
  - conversational
  - experimental
---

# Qwen3.8-27B MXFP4 MxWave GGUF

This repository packages
[`kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave`](https://huggingface.co/kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave)
as GGUF for recent llama.cpp and Ollama runtimes.

The package contains:

- a language-model GGUF that preserves the source checkpoint's 400 MXFP4
  projections and embeds its 15 BF16 NextN/MTP tensors;
- a separate BF16 vision projector;
- the tokenizer and chat template embedded in the main GGUF;
- tested text generation, thinking, tool calling, vision, and embedded-MTP
  support on one NVIDIA DGX Spark.

The GGUF conversion does not quantize the model again. It losslessly repacks
the existing MXFP4 codes and E8M0 scales, and applies the byte-layout and
Qwen3.8 linear-attention V-head permutation required by llama.cpp.

This is an experimental community conversion. It is not an official Qwen,
NVIDIA, Ollama, or ggml-org release.

## What the comparison showed

On one DGX Spark with the pinned Ollama configuration below:

- with speculative decoding disabled, this artifact delivered **18.9% higher
  decode throughput** and **4.4% higher approximately-64K prefill throughput**
  than the official ggml-org Q4_K_M conversion;
- with MTP-2 enabled on both packages, official Q4_K_M plus its separate Q4_0
  MTP sidecar delivered **8.1% higher normalized decode throughput**;
- this package includes embedded MTP and is **1.61 GiB smaller** than the
  corresponding official MTP-enabled package;
- the limited quality checks found no detectable regression, but they are not
  sufficient to claim that either quantization has better overall quality.

In short, this is a competitive, storage-efficient Ollama/DGX Spark package,
especially for normal non-speculative inference. It is not claimed to be the
universally fastest or highest-quality Qwen3.8 GGUF.

## Model details

| Property | Value |
|---|---|
| Original model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Original revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Source MXFP4 checkpoint | [`kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave`](https://huggingface.co/kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave) |
| Source checkpoint revision | `c0ffe6b8e5b099bf490395c047c8cbea021b100e` |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Quantized projections | 400 language-model projections |
| Stored weight format | OCP MXFP4 E2M1 with E8M0 scales and 32-value blocks |
| GGUF conversion | Lossless repack of the source quantized checkpoint |
| Vision | Separate BF16 projector |
| MTP | 15 BF16 NextN/MTP tensors embedded in the main GGUF |
| Native context metadata | 262,144 tokens |
| Tested context | 65,536 tokens |
| Tested hardware | One NVIDIA DGX Spark, GB10/SM121 |
| Tested Ollama | `0.35.1-rc2` |
| Tested llama.cpp | tag `b11232`, commit `6f767fe960c3b97cf37fac4626c86400561ca1e4` |
| MxWave GGUF exporter | PR-head commit [`cb547eaa`](https://github.com/kirillbilchenko/MxWave/commit/cb547eaa6d4a2fe46d15d53300e2aa6c6c2d4d17), merged as [`54af8158`](https://github.com/kirillbilchenko/MxWave/commit/54af81589bed9d8f3a8f9f28c95907ab07e69ea9) |

`H64` in the filename identifies the source checkpoint's 64-sequence
block-Hessian calibration recipe. It is not a precision name.

## Files and checksums

| File | Role | Bytes | SHA-256 |
|---|---|---:|---|
| `qwen3.8-27b-mxwave-h64-mxfp4.gguf` | Language model with embedded MTP | 18,927,519,584 | `77d502a587a484b6dbe8002b270f1ea228772e476c6b887a51eff5347b7e0afd` |
| `mmproj-qwen3.8-27b-mxwave-h64-bf16.gguf` | BF16 vision projector | 931,146,304 | `9d772c5bad471017c3bfdaf41d0eba66816a39a92a9fb772c3e09798b1478ccb` |

The combined GGUF payload is 19,858,665,888 bytes, approximately **18.495
GiB**. Verify it after download:

```bash
sha256sum -c SHA256SUMS
```

Do not combine the main GGUF with a projector from another conversion.

## Compatibility

This artifact needs a recent llama.cpp/Ollama stack with Qwen3.8 hybrid-model,
MXFP4 GGUF, multimodal-projector, and embedded-MTP support. The tested runtime
was Ollama `0.35.1-rc2`. Older versions may reject the model, omit vision, or
fail to activate MTP.

The measured runtime used Q8 matrix-multiplication precision:

```bash
GGML_CUDA_MMQ_PREC=q8
```

Changing runtime precision may change both throughput and output quality.
Compatibility with other GGUF applications has not been systematically tested.

## Download

```bash
HF_XET_HIGH_PERFORMANCE=1 hf download \
  kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave-GGUF \
  --include "*.gguf" \
  --include "Modelfile*" \
  --include "SHA256SUMS" \
  --local-dir ./Qwen3.8-27B-MXFP4-MxWave-GGUF

cd Qwen3.8-27B-MXFP4-MxWave-GGUF
sha256sum -c SHA256SUMS
```

The tested Ollama import copied the GGUFs into its blob store. Budget roughly
another 18.5 GiB if you retain both the downloaded files and the Ollama copy.

## Ollama usage

Use an explicit Modelfile so that the vision projector and MTP policy are
unambiguous. A bare `ollama run hf.co/...` is not the recommended path for this
two-file package.

### Recommended DGX Spark mode: embedded MTP-2

The included `Modelfile` contains:

```dockerfile
FROM ./qwen3.8-27b-mxwave-h64-mxfp4.gguf
FROM ./mmproj-qwen3.8-27b-mxwave-h64-bf16.gguf
PARAMETER num_ctx 65536
PARAMETER draft_num_predict 2
```

Import and run it:

```bash
ollama create qwen3.8-27b-mxwave:mtp2 -f Modelfile
ollama run qwen3.8-27b-mxwave:mtp2
```

No separate `DRAFT` file is needed because the MTP tensors are embedded in the
main GGUF.

### Reproducible baseline: MTP disabled

```bash
ollama create qwen3.8-27b-mxwave:baseline -f Modelfile.baseline
ollama run qwen3.8-27b-mxwave:baseline
```

The controlled measurements used:

```bash
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_KEEP_ALIVE=-1 \
GGML_CUDA_MMQ_PREC=q8 \
ollama serve
```

If Ollama already runs as a service, configure the variables on that service
instead of starting a second server.

### OpenAI-compatible API

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-mxwave:mtp2",
    "messages": [
      {"role": "user", "content": "Write a small Python retry helper."}
    ],
    "stream": false
  }'
```

## Controlled comparison with official Q4_K_M

The comparator was
[`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
at revision `71bc7b627595dc8a91039addd9c791ae548d6747`.

### Throughput

| Measurement | MxWave GGUF | Official Q4_K_M | Result |
|---|---:|---:|---|
| Baseline decode, MTP disabled | **13.9136 tok/s** | 11.7005 tok/s | MxWave +18.9% |
| Unique approximately-64K prefill | **888.666 tok/s** | 851.328 tok/s | MxWave +4.4% |
| MTP-2 decode | 19.0802 tok/s | **20.6332 tok/s** | Official +8.1% |
| MTP-2 median task wall time | 22.37 s | 22.48 s | Effectively tied¹ |

¹ The MxWave and official MTP responses contained 421 and 457 generated tokens,
respectively. Do not interpret the wall-time result as normalized throughput.

### Package size

Package size includes the main model and BF16 projector. The official optimized
configuration additionally needs its separate Q4_0 MTP sidecar.

| Configuration | Package size |
|---|---:|
| MxWave, including embedded MTP | **18.495 GiB** |
| Official Q4_K_M baseline | 18.538 GiB |
| Official Q4_K_M plus Q4_0 MTP sidecar | 20.103 GiB |

### Quality checks

| Check | MxWave | Official Q4_K_M |
|---|---:|---:|
| Deterministic qualification suite | 9/10 | 9/10 |
| Corrected unique-marker 64K retrieval | 2/2 | 2/2 |
| Tool-calling smoke case | Pass | Pass |
| Hidden executable Python tests | Pass | Pass |
| Vision smoke case | Pass | Pass |
| GSM8K directional pilot | 19/25 | 18/25 |

Both models failed the same original 64K fixture involving the expected value
`violet-931`, then both passed two follow-up 64K cases using unique alphanumeric
markers. The original result is therefore treated as fixture sensitivity, not
evidence differentiating the quantizations.

The GSM8K pilot used only the first 25 examples. A one-answer difference is not
statistically meaningful. This supports “no detected regression,” not “better
quality.” Vision and tool calling were functional smoke tests, not broad quality
benchmarks.

## Benchmark protocol

Measurements were recorded on 2026-10-02 using one DGX Spark. Common settings:

- Ollama `0.35.1-rc2`;
- context length 65,536;
- Q8 runtime matrix-multiplication precision;
- flash attention and one parallel sequence;
- hidden reasoning disabled;
- identical prompts and settings for both artifacts.

For throughput, two balanced rounds were run in A/B then B/A order. Every block
unloaded the prior model, excluded one cold warm-up, ran two sustained decode
repetitions, and ran one unique approximately-64K prefill/retrieval request. The
prefill requests reported zero cached prompt tokens. Results are medians of four
decode samples and two prefill samples per model.

The fixed decode prompt was:

> Write exactly 20 numbered, concise rules for reviewing a Python pull request
> that adds an LLM inference backend. Use one sentence per rule; cover
> correctness, API compatibility, security, resource limits, observability,
> testing, and rollback. Do not add a preface or conclusion.

Raw measurements and pilot configurations are in `evaluation/`.

## Conversion provenance

The GGUF exporter is part of
[`kirillbilchenko/MxWave`](https://github.com/kirillbilchenko/MxWave) and was
introduced by [PR #13](https://github.com/kirillbilchenko/MxWave/pull/13).

Pinned conversion inputs:

- source MxWave checkpoint revision: `c0ffe6b8e5b099bf490395c047c8cbea021b100e`;
- MxWave GGUF exporter PR-head commit: `cb547eaa6d4a2fe46d15d53300e2aa6c6c2d4d17`;
- merged MxWave commit: `54af81589bed9d8f3a8f9f28c95907ab07e69ea9`;
- llama.cpp commit: `6f767fe960c3b97cf37fac4626c86400561ca1e4`;
- Ollama image: `ollama/ollama:0.35.1-rc2`;
- tested Ollama ARM64 image digest:
  `sha256:00af1106f8cf178e4054270d145e7bfef82b38896d9339a4335047872faa0226`.

The artifacts were structurally verified before publication. See
`export-manifest.json` and `publication-manifest.json` for machine-readable
provenance.

## Intended use

Suitable uses include local coding and research assistance, evaluation of
MXFP4 GGUF, single-user Ollama deployment on a DGX Spark, embedded-MTP
experimentation, and local multimodal experiments. Evaluate the model on your
own workloads before production use.

## Limitations

- Performance results apply only to the tested DGX Spark and pinned runtime.
- Only 65,536-token operation was tested here; the card does not claim that the
  larger native context is practical or qualified on one Spark.
- GSM8K used only 25 examples and is directional evidence.
- Vision, tool calling, coding, and retrieval coverage establish compatibility,
  not broad task quality.
- Official Q4_K_M plus its separate MTP sidecar was faster in normalized MTP-2
  decode throughput.
- The published comparison used Q8 runtime precision, not the experimental Q4
  runtime mode.
- Quantization can change token probabilities and outputs even when small
  aggregate tests appear similar.
- This conversion adds no safety training and inherits the behavior, risks, and
  limitations of the original model.

## License and attribution

The model artifacts inherit the
[Apache License 2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE)
from Qwen3.8-27B. MxWave and its GGUF exporter are also Apache-2.0.

Redistributions should retain `LICENSE`, attribution, modification notices, and
links to the original model and source MXFP4 checkpoint. Apache-2.0 does not
grant rights to upstream trademarks. Consult the
[original Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) for
its intended uses and limitations.
