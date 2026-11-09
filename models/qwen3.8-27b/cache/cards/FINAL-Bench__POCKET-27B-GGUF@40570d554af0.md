---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- ko
- en
tags:
- gguf
- llama.cpp
- quantized
- conversational
- multimodal
- vision-language
- local-llm
- on-device
- pocket
- vidraft
- archsloth
- qwen3_5
- 27b
---

<!-- POCKET-FAMILY-NOTE:START -->
**From the VIDRAFT POCKET family.** This 27B package sits alongside [POCKET-35B](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF), [POCKET-26B](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) and [POCKET-Darwin-180B](https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF). Family benchmark news, including the Darwin-180B SuperCLUE result, belongs to the model named in its source card. It is **not a result for POCKET-27B**. This package does not apply Darwin RSI or ARMORING training.
<!-- POCKET-FAMILY-NOTE:END -->

# POCKET-27B-GGUF

ARCHsloth's Qwen3.8-27B Q6_K, distributed with unchanged model weights in the POCKET family. Text chat uses one 22.43 GB file; vision additionally needs the separate 1.84 GB projector.

![POCKET-27B overview: unchanged ARCHsloth Q6_K weights, a 22.43 GB model file, and optional vision with a separate projector](./pocket-overview.svg)

[Download files](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF/tree/main) · [Download and run](#download-and-run) · [Benchmarks and limits](#quality-and-evaluation)

[POCKET-35B](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) · [POCKET-26B](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) · [POCKET-27B](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF) · [POCKET-Darwin-180B](https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF)

## At a glance

| Item | Details |
|---|---|
| Artifact | Qwen3.8-27B-family dense vision-language model; standard Q6_K GGUF |
| Original checkpoint | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Immediate distribution source | [Archsloth/Qwen3.8-27B-GGUF](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/tree/87b01a847b03c2c4e979453371a8fc6c2d65fe7b), revision `87b01a847b03c2c4e979453371a8fc6c2d65fe7b` |
| Supplied model build | `Qwen3.8-27B-AS-Q6_K.gguf`, 22,430,999,840 bytes |
| Optional vision component | `mmproj-F32.gguf`, 1,842,940,064 bytes; the source's full-precision vision tower |
| Hardware | Sufficient working memory beyond the selected files; actual memory and speed depend on runtime, context and device |
| Runtime | llama.cpp with support for this Qwen3.8-family GGUF and, for images, its matching projector; check the installed build |
| License | Apache-2.0, as declared by the source and original checkpoint; retain attribution |
| Evidence status | Existing ARCHsloth publisher measurements, attributed below; no new CPU benchmark or device-fit measurement for this release |

**What changed for POCKET:** distribution, model-card organization, overview artwork and family navigation. The Q6_K model and F32 projector are redistributed unchanged from the pinned ARCHsloth source. No new training, pruning, quantization, rounding search, ARMORING or RSI is applied by this repackaging. The existing filenames are retained.

## Choose your build

| Build or component | Exact file size | Memory guidance | Best for |
|---|---:|---|---|
| [Q6_K model](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF/resolve/main/Qwen3.8-27B-AS-Q6_K.gguf) | 22,430,999,840 bytes, about 22.43 GB or 20.89 GiB | Allow additional OS, runtime, context and KV-cache memory; no reliable device-fit threshold is established here | Text-only generation; also the required language-model file for vision |
| [F32 vision projector](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF/resolve/main/mmproj-F32.gguf) | 1,842,940,064 bytes, about 1.84 GB or 1.72 GiB | Additional component and runtime memory; not a standalone language model | Image input with the matching Q6_K model and a supporting multimodal runtime |

Text-only use needs only the first file. Vision needs both, totaling 24,273,939,904 bytes, about 24.27 GB, **before runtime memory**. File sizes are not peak RAM or VRAM measurements. A 22.43 GB model file does not, by itself, establish fit on a particular GPU, laptop or fixed process-memory budget.

This repository supplies one model quantization: Q6_K. It does not offer a POCKET Q4 or Q2 build. The original publisher's Q4 candidate and its English-quality failure are discussed below rather than presented as a downloadable success.

## Download and run

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and a [llama.cpp build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) supporting the selected model and projector. Download only the files needed for the intended mode, not the whole repository.

**1. Download the language model.**

```bash
hf download FINAL-Bench/POCKET-27B-GGUF Qwen3.8-27B-AS-Q6_K.gguf --revision d529cd73fea869b5f74545314270ef8885692964 --local-dir ./POCKET-27B
```

**2. Illustrative CPU-only text command.**

```bash
llama-cli -m ./POCKET-27B/Qwen3.8-27B-AS-Q6_K.gguf --device none -ngl 0 -t 8 -c 2048 -n 256 -p "Explain why local AI can be useful."
```

**3. Optional vision component and illustrative local server.**

```bash
hf download FINAL-Bench/POCKET-27B-GGUF mmproj-F32.gguf --revision d529cd73fea869b5f74545314270ef8885692964 --local-dir ./POCKET-27B
llama-server -m ./POCKET-27B/Qwen3.8-27B-AS-Q6_K.gguf --mmproj ./POCKET-27B/mmproj-F32.gguf --jinja -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

These are **unexecuted examples for this POCKET release**, not successful-run receipts, benchmarks or memory-fit guarantees. The CPU example requests no GPU device; the server example requests GPU layer offload, which is not proof of the placement actually achieved. Check runtime logs and memory on the target device. The context values and eight CPU threads are illustrative, not measured optima; start with a thread count appropriate to the physical CPU cores.

The CLI is for text input. Vision requires the separate projector plus an image-capable client or the supporting server interface. Do not load the projector as the language model. Commands use standalone llama.cpp executable names; distributions using a unified launcher should consult their own help. No cloud-inference endpoint or website login is required by these examples.

## Runtime compatibility

| Runtime or application | What must be checked | Evidence and limits |
|---|---|---|
| llama.cpp, text | Support for the exact Qwen3.8-family GGUF and standard Q6_K tensors | The source card documents upstream llama.cpp use; no universal minimum version or new execution result is asserted here |
| llama.cpp, vision | Matching `mmproj-F32.gguf`, multimodal model support and sufficient memory | The source provides a vision command, but its vision-quality evaluation is listed as pending |
| Ollama, LM Studio and other GGUF applications | Application version, bundled runtime, exact model architecture and projector support | A GGUF file is not a guarantee of compatibility; this release does not supply a tested app/version matrix |
| CPU, GPU and other accelerators | The installed backend, supported operations, actual placement and resource use | No new device benchmark is included; neither the file format nor an offload flag proves GPU or NPU execution |

See [llama.cpp model-loading guidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md). If loading fails, preserve the error and runtime version rather than treating failure as a quality score or silently substituting another model.

## Hardware and performance

No new CPU benchmark, GPU benchmark, NPU run, peak-memory measurement or device-fit validation was performed for this POCKET repackaging. The existing Q6_K file is a **dense** model; the sparse-MoE speeds and active-parameter counts of POCKET-35B or POCKET-Darwin-180B do not transfer to it.

| Device or mode | Configuration | Generation | Evidence and limits |
|---|---|---|---|
| CPU-only text | Q6_K model, supported runtime, adequate RAM and context headroom | Not measured for this release | An illustrative command is supplied, not a run receipt |
| GPU text or vision | Supported backend; projector also required for vision | Not measured for this release | Requested offload is not verified placement or a speed promise |
| NPU | No validated inference path supplied | Not measured | No NPU success is claimed |
| Phone, Mac, laptop or a fixed process-memory budget | No corresponding peak-memory/context trial supplied | Not measured | File size alone does not establish fit or usable speed |

Report the exact revision, file, runtime/build, device, power mode, context, thread and batch settings, peak memory, prompt-processing rate and generation rate when sharing a device result. Keep warmup and measurement samples separate. A hosted demo or another family member's benchmark is not a measurement of this package.

## Quality and evaluation

The following are **measurements reported by the original ARCHsloth publisher**, not new POCKET measurements or independent reproduction. They apply to the unchanged Q6_K artifact from the pinned source. The source compares KL divergence from the BF16 original using `llama-perplexity --kl-divergence`, context 512, 60 chunks, and held-out Korean and English evaluation text separate from calibration. Lower KL divergence is better; it is not a task-accuracy percentage.

| Original publisher's artifact | File bytes | Korean KL | English KL |
|---|---:|---:|---:|
| ARCHsloth Q6_K, redistributed here | 22,430,999,840 | 0.001385 | 0.001254 |
| unsloth UD-Q6_K reference | 21,983,677,344 | 0.002195 | 0.001476 |
| unsloth UD-Q6_K_M reference | 23,088,409,504 | 0.001632 | 0.001137 |

Against the same-rung UD-Q6_K reference, the source reports 36.9% lower Korean KL and 15.0% lower English KL. Against the larger UD-Q6_K_M reference, the tradeoff is mixed: 15.1% lower Korean KL but 10.3% higher English KL. These percentages describe the reported KL comparison, **not an equivalent percentage improvement in intelligence, benchmark accuracy or token speed**.

**Preserved failed candidate:** the source reports that its Q4_K_M candidate had 12.8% lower Korean KL but **15.8% higher English KL** than the compared unsloth UD-Q4_K_M file, so that candidate was not shipped. It is not a successful POCKET Q4 build and is not supplied here.

The source measured two language axes, not its separate 4B model's ten-axis sweep. Long-context retrieval, agentic tool use and vision evaluation remain pending in the source evidence. There is no POCKET-27B GPQA, SuperCLUE, general-quality-equivalence or universally lossless-quantization result in this card. See the pinned [source README](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/blob/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/README.md), [EVAL.md](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/blob/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/EVAL.md) and [per-run logs](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/tree/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/eval/logs) for the publisher's protocol and evidence.

<details>
<summary>Lineage and what the original quantization changed</summary>

The source identifies Qwen/Qwen3.8-27B as the original checkpoint, standard llama.cpp Q6_K as the tensor layout, and Intel AutoRound with its SignRoundV2 extension as the rounding-search tooling. Its Korean/English calibration corpus and evaluation materials are included in the source repository. The F32 vision tower was carried at full precision.

Those are the **original ARCHsloth quantization's methods**, not new steps performed for this POCKET distribution. This is not an ARMORING-trained or Darwin RSI checkpoint. Read the source's [CALIBRATION.md](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/blob/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/CALIBRATION.md) for the original calibration description.

</details>

## Intended uses

- Local text-generation experiments with a supported runtime and adequate memory.
- Korean and English workflows evaluated against a task-specific baseline, with the original KL results used only within their stated scope.
- Optional image-input experiments using the supplied matching projector, without assuming validated vision quality.
- Reproducible comparisons that identify the exact file and distinguish unchanged repackaging from new quantization or training.

## Limitations

- Repackaging does not reduce the existing Q6_K file size, create a new quantization or establish a speed improvement.
- No new CPU/GPU/NPU inference, peak-memory or device-fit measurement is supplied for this release.
- Model file size is not runtime memory; fit within a fixed memory budget, phone or particular GPU is unverified.
- Source KL measurements cover two language axes, not general task accuracy, long-context retrieval, tools or vision.
- The source's Q4 English degradation remains a failed candidate, not a hidden successful build.
- The dense 27B model is distinct from the family's Gemma4 and sparse-MoE artifacts. Their parameter activity, speeds, scores and pruning methods do not transfer.
- No ARMORING, RSI, additional training, expert pruning, smaller-bit optimization or new rounding search is applied by this distribution.
- Text generation may be incorrect or unsafe; suitability requires task-specific evaluation and appropriate human oversight.

## License and attribution

The [source model card](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/blob/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/README.md#license) and [original Qwen checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B) declare **Apache-2.0**. Read the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0), retain applicable notices and attribution, and review the source materials when redistributing. This paragraph is attribution, not a separate legal clearance.

Original model: Qwen team / Alibaba. Original Q6_K rounding optimization, calibration and reported measurements: ARCHsloth. Runtime and GGUF tooling: llama.cpp contributors. Rounding-search tooling identified by the source: Intel AutoRound. POCKET repackaging, overview and this organized model card: VIDRAFT / FINAL-Bench. The ARCHsloth artifact filenames and source provenance are retained rather than attributing its quantization work to a new POCKET run.

## Support and updates

For this distribution, use the [model files](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF/tree/main) and [model discussions](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF/discussions). For original quantization methodology and measurements, follow the pinned ARCHsloth references below. Include the revision, filename, runtime version/build, device, context, memory and exact command when reporting a problem; do not post private prompts or credentials.

This card does not assert that a POCKET-27B hosted demo or model-specific website setup page exists. Direct model-file access does not require an external website login. Future changes must keep existing source attribution and clearly distinguish new artifacts or measurements from this unchanged-source distribution.

## POCKET family

| Model | Role | Start here |
|---|---|---|
| POCKET-35B-GGUF | Full 35B sparse-MoE PC builds; Q2_K recommended in its own card | [Model card](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) |
| POCKET-26B-GGUF | Korean-tuned Gemma4 GGUF; 11 GB and 17 GB options | [Model card](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) |
| POCKET-27B-GGUF | Unchanged ARCHsloth Qwen3.8-27B Q6_K; separate optional F32 vision projector | [Model card](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF) |
| POCKET-Darwin-180B-GGUF | Text-only RSI reasoning; four required shards | [Model card](https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF) |

Explore the [POCKET collection](https://huggingface.co/collections/FINAL-Bench/pocket-models-6a618ee5d23eafb7e185a5c6). Each package has its own model lineage, quantization, license, memory requirements and evidence; membership is not a shared performance or quality guarantee.

<details>
<summary>More POCKET projects and research</summary>

Phone-oriented derivatives have separate requirements: [KR GGUF](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF), [KR MLX](https://huggingface.co/FINAL-Bench/POCKET-KR-MLX) and [EN GGUF](https://huggingface.co/FINAL-Bench/POCKET-EN-GGUF). The [Qwen3.8 Flash-Next GGUF](https://huggingface.co/FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF) is a separate parent-model package, not this dense 27B model or the Darwin-180B RSI-R3 weights. Image-generation projects are also separate: [POCKET-Image-Zimage](https://huggingface.co/FINAL-Bench/POCKET-Image-Zimage) and [POCKET-Zimage-CPU](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU).

</details>

## References

- [Immediate source and publisher measurements, pinned revision 87b01a84](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/blob/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/README.md).
- [Original source files at that revision](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/tree/87b01a847b03c2c4e979453371a8fc6c2d65fe7b), [evaluation notes](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/blob/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/EVAL.md), [calibration notes](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/blob/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/CALIBRATION.md) and [raw evaluation logs](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF/tree/87b01a847b03c2c4e979453371a8fc6c2d65fe7b/eval/logs).
- [Original Qwen3.8-27B checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B).
- [Hugging Face CLI download documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).
- [llama.cpp build and CPU-backend guidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#notes-about-gpu-accelerated-backends), [model loading](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md) and [multimodal tooling](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd).
- [Intel AutoRound](https://github.com/intel/auto-round), identified as original rounding-search tooling by the source.
- Existing POCKET structure: [35B](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF), [26B](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) and [Darwin-180B](https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF).

Documentation prepared on 2026-11-09. Source publisher measurements are not relabeled as newly reproduced results.
