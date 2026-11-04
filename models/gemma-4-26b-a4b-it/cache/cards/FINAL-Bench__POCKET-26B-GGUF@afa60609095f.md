---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
- google/gemma-4-26B-A4B-it
tags:
- gguf
- llama.cpp
- conversational
- on-device
- mobile
- korean
- korean-llm
- cpu
- local-llm
- edge
- gemma
- gemma4
- mixture-of-experts
- moe
- pocket
- vidraft
---

# POCKET-26B-GGUF

Korean-tuned Gemma4 GGUF for local text assistance, with 11 GB and 17 GB build options.

![POCKET-26B overview: Korean-tuned Gemma4, Q2_K recommended, CPU-only option](./pocket-overview.svg)

[Download files](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF/tree/main) · [Setup guide](https://vidraft.net/p/pocket.html?model=pocket_26b&utm_source=huggingface&utm_medium=referral&utm_campaign=pocket_2026q4&utm_content=pocket_26b_model_card_top) · [Benchmarks](#quality-and-evaluation)

[POCKET-35B](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) · [POCKET-26B](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) · [POCKET-Darwin-180B](https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF)

## At a glance

| Item | Details |
|---|---|
| Artifact | Gemma4-derived 25.2B-parameter MoE, approximately 4B active parameters per token; GGUF |
| Starting checkpoint | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Recommended first build | Q2_K, approximately 11 GB; mixed-precision quantization |
| Hardware | A device with sufficient working memory beyond the file size; a CPU-only execution path is available |
| Runtime | llama.cpp with Gemma4 support; exact app/version compatibility must be checked |
| License | Apache-2.0; see [License and attribution](#license-and-attribution) |
| Evidence status | Publisher measurements with device-specific conditions below |

This package keeps the source model unpruned and applies VIDRAFT's Korean-tuned quantization. It is not a promise of universal app support, a measured 12 GB-phone fit, or an MLX package.

## Choose your build

| Build | File size | Memory guidance | Best for |
|---|---|---|---|
| [Q2_K — recommended](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF/resolve/main/POCKET-26B-Q2_K.gguf) | About 11 GB | Peak RAM and phone-fit measurements not supplied; allow OS, runtime and context headroom | Smaller download with the reported GPQA result below |
| [Q4_K_M](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF/resolve/main/POCKET-26B-Q4_K_M.gguf) | About 17 GB | A higher-memory device; actual peak depends on context and backend | Higher-precision alternative |

File sizes are rounded decimal GB, not process-memory measurements. An 11 GB file does **not** establish reliable operation on a 12 GB phone: the operating system, context, runtime buffers and other applications also need memory. For smaller phone-oriented artifacts, see the separate [POCKET-KR-GGUF](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF) card and its limits.

## Download and run

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and a [llama.cpp build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) that supports Gemma4. The example downloads only the recommended file from the audited revision; it does not clone or download the whole repository.

**1. Download the recommended build.**

```bash
hf download FINAL-Bench/POCKET-26B-GGUF POCKET-26B-Q2_K.gguf --revision 9f49f5c276efafe909f6b2b2fb161181ce417a45 --local-dir ./POCKET-26B
```

**2. Run on the CPU.**

```bash
llama-cli -m ./POCKET-26B/POCKET-26B-Q2_K.gguf --device none -ngl 0 -t 8 -c 4096 -n 256 -p "Explain why local AI can be useful."
```

The expected result is generated text in the terminal, not a fixed benchmark answer. These command examples were syntax-reviewed but not executed during this review.

Commands use the standalone llama.cpp executable names. Some distributions expose a unified `llama` launcher; consult that distribution's help and the [official model-loading documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md). Use a physical-core-based thread count appropriate to your device; a higher number is not automatically faster.

For installation, memory errors and app-specific help, use the [model-specific setup guide](https://vidraft.net/p/pocket.html?model=pocket_26b&utm_source=huggingface&utm_medium=referral&utm_campaign=pocket_2026q4&utm_content=pocket_26b_quickstart_help). Model files remain directly available from Hugging Face without a VIDRAFT login.

## Runtime compatibility

| Runtime | Requirement | Evidence and limits |
|---|---|---|
| llama.cpp | A build with Gemma4 support | The original card reports upstream use; no exact universal minimum version was supplied |
| Ollama | A version supporting Gemma4 and the selected GGUF | Generic [Hugging Face GGUF integration](https://huggingface.co/docs/hub/ollama) exists; an exact file/version compatibility matrix is not supplied |
| LM Studio | A bundled runtime supporting Gemma4 | Paste this repository URL into model search, select a file and check the app's memory guidance |
| PocketPal, koboldcpp and browser runtimes | Explicit Gemma4 support plus sufficient memory | No universal app or phone compatibility claim is made |
| Apple MLX | A separately supplied MLX artifact | These downloads are GGUF files, not native MLX weights |

The same file format does not guarantee support in every app or version. A dated, exact-version compatibility matrix remains a community contribution opportunity.

## Hardware and performance

These are the limits of the supplied device evidence, not estimated performance promises. The CPU-only quickstart explicitly disables GPU acceleration with `--device none`; `-ngl 0` alone is not the strongest no-GPU guarantee in every backend.

| Device or mode | Configuration | Generation | Evidence and limits |
|---|---|---|---|
| PC CPU | Selected GGUF; enough RAM for the actual context | No throughput figure supplied | A CPU execution path is documented; no device benchmark is supplied |
| Hosted CPU demo | [POCKET-26B-CPU](https://huggingface.co/spaces/FINAL-Bench/POCKET-26B-CPU) | No throughput figure supplied | A hosted service, not a local-device benchmark |
| iPhone / Mac | No matching throughput measurement supplied | Not measured | Community reports welcome |
| Phone with 12 GB RAM | No matching peak-memory/context measurement supplied | Not measured | File size alone does not establish fit or usable speed |

A useful community result identifies the exact file, runtime version, device, context size, peak memory and prompt/generation rates. Do not transfer POCKET-35B's Xeon or Mac speeds to this different architecture.

## Quality and evaluation

Publisher-reported GPQA Diamond results: a single greedy pass over 198 questions using the publisher's harness.

| Build | Evaluation | Result | Conditions and limits |
|---|---|---|---|
| Gemma4-26B-A4B reference | GPQA Diamond | 67.7% | Reported base comparison, 198 questions, greedy |
| POCKET-26B Q4_K_M | GPQA Diamond | 67.7% | Same reported score on this evaluation |
| POCKET-26B Q2_K | GPQA Diamond | 67.2% | About 0.5 percentage points lower, approximately one question |

This small comparison does **not** establish general statistical equivalence or universally lossless quantization. Paired answer-level results, a prespecified equivalence margin, full harness details and broader tasks would be needed for a stronger equivalence claim. No dedicated Korean task-quality score is supplied in the source card; the GPQA table should not be presented as a Korean-language quality benchmark.

<details>
<summary>Quantization and unpruned-model notes</summary>

The original card describes Korean-tuned mixed precision: more sensitive weights remain at higher precision. It reports that the model remains unpruned, retaining all 128 experts, and that pruning harmed its low-bit robustness.

The original card also mentions approximately 44% for a plain uniform-Q2 comparison. The exact comparator file and full evaluation configuration were not supplied, so this is a publisher-reported observation, not an independently verified ranking or an equivalent-format guarantee.

</details>

## Intended uses

- Local text assistance and Korean-language workflow experimentation on suitably provisioned devices.
- Comparing the 11 GB and 17 GB quantization options with a task-specific evaluation.
- CPU-only workflows where local execution is preferred; no cloud service is required by the supplied command.

## Limitations

- Exact peak memory, reliable phone fit and on-device iPhone/Mac throughput were not supplied.
- “GGUF” does not mean every app or version supports Gemma4.
- The GPQA comparison is one 198-item greedy evaluation, not general quality equivalence.
- Korean-tuned quantization is the package's positioning; a dedicated Korean task benchmark is not supplied here.
- File size is not a complete RAM budget; longer context increases memory requirements.

## License and attribution

This model card declares **Apache-2.0**, consistent with the current [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) metadata and Google's [official Gemma 4 license page](https://ai.google.dev/gemma/apache_2).

Starting checkpoint: Gemma4-26B-A4B-it by Google. Korean-tuned quantization and this package: VIDRAFT / FINAL-Bench. Runtime tooling: the llama.cpp contributors. Read the license and retain required notices and attribution when redistributing.

## Support and updates

For device selection and installation help, visit the [VIDRAFT POCKET setup guide](https://vidraft.net/p/pocket.html?model=pocket_26b&utm_source=huggingface&utm_medium=referral&utm_campaign=pocket_2026q4&utm_content=pocket_26b_model_card_footer). Report reproducible issues in the [model discussions](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF/discussions), including the filename, runtime version, device, memory and command; do not post private prompts or credentials.

A browser-accessible [POCKET-26B CPU demo](https://huggingface.co/spaces/FINAL-Bench/POCKET-26B-CPU) is a separate hosted service; its availability and latency are not local-device measurements. No local-inference telemetry or website gate is required to obtain these public files.

## POCKET family

| Model | Role | Start here |
|---|---|---|
| POCKET-35B-GGUF | Full 35B PC builds; Q2_K recommended first | [Model card](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) |
| POCKET-26B-GGUF | Korean-tuned Gemma4 GGUF; 11 GB and 17 GB options | [Model card](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) |
| POCKET-Darwin-180B-GGUF | Text-only RSI reasoning; four-shard download | [Model card](https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF) |

Explore the [POCKET collection](https://huggingface.co/collections/FINAL-Bench/pocket-models-6a618ee5d23eafb7e185a5c6). For phone-oriented derivatives, see [KR GGUF](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF), [KR MLX](https://huggingface.co/FINAL-Bench/POCKET-KR-MLX) and [EN GGUF](https://huggingface.co/FINAL-Bench/POCKET-EN-GGUF).

<details>
<summary>More POCKET projects and research</summary>

[Qwen3.8 Flash-Next GGUF](https://huggingface.co/FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF) is a separate parent-model package, distinct from the POCKET-Darwin-180B RSI-R3 weights. Image projects are also separate: [POCKET-Image-Zimage](https://huggingface.co/FINAL-Bench/POCKET-Image-Zimage), [POCKET-Zimage-CPU](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU) and [Image Studio](https://huggingface.co/spaces/FINAL-Bench/POCKET-Image-Studio).

Research collections: [Darwin](https://huggingface.co/collections/FINAL-Bench/darwin-family-699987b1f652864af0122193), [Aether](https://huggingface.co/collections/FINAL-Bench/aether-foundation-model-6a5c7f2fa1a4165c0414e53a), [VKAE](https://huggingface.co/collections/FINAL-Bench/vkae-accelerated-6a47231d7e7999dd8227675a).

</details>

## References

- [Source card and publisher measurements, revision 9f49f5c](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF/blob/9f49f5c276efafe909f6b2b2fb161181ce417a45/README.md).
- [Google's Gemma4 starting checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it) and [Apache-2.0 license](https://ai.google.dev/gemma/apache_2).
- [Hugging Face CLI download documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).
- [llama.cpp build and CPU-backend guidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#notes-about-gpu-accelerated-backends).
- [llama.cpp model loading](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md).
- [Hugging Face with Ollama](https://huggingface.co/docs/hub/ollama) and [LM Studio](https://huggingface.co/docs/hub/lmstudio).
- [VIDRAFT: on-device LLMs without a GPU](https://vidraft.net/insights/on-device-llm-without-gpu.html) and [model quantization](https://vidraft.net/insights/what-is-quantization-llm.html).

Documentation reviewed on 2026-11-04.
