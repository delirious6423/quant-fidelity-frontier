---
base_model:
- ukisai/Swift1.5-Qwen3.8-Flash-Next
- Qwen/Qwen3.8-Flash-Next
tags:
- dwarfstart4
- antirez
- ds4
- gguf
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/blob/main/LICENSE
---

# Swift 1.5 Qwen3.8-Flash-Next for DwarfStar 4

This is a single-file GGUF conversion of UkisAI’s Swift 1.5 for local inference with DwarfStar 4 (`ds4`). UkisAI post-trained Qwen3.8-Flash-Next to shorten reasoning traces while preserving task performance. The DS4 conversion packages mixed-precision weights, an embedded multi-token prediction (MTP) layer, and a disk-backed n-gram table. [UkisAI model card](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next) · [DwarfStar 4 Qwen guide](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)

## Model Details

### Model Description

- **Base model** — Qwen developed Qwen3.8-Flash-Next.
- **Swift 1.5 post-training** — UkisAI developed and released the Swift checkpoint. UkisAI credits NVIDIA Innovation Lab, AWS, and Google Cloud for compute and infrastructure support. [UkisAI acknowledgements](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#acknowledgements)
- **DS4 conversion** — The maintainer recalls preparing the conversion using antirez’s DwarfStar 4 guidance. The local GGUF and its manifest record the resulting build. The exact conversion commands and runtime commit are unconfirmed.
- **Model type** — Autoregressive mixture-of-experts language model with a separate vision projector for compatible runtimes.
- **Languages** — The upstream evaluation includes English and Chinese benchmarks. Per-language results for this DS4 build are unavailable. [UkisAI evaluation](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#evaluation)
- **Weight licenses** — Swift Open License v1.0 applies to UkisAI’s contribution. Qwen Community License 1.0 applies to the underlying Qwen model. Review both licenses and the upstream attribution notice for redistribution or commercial use. [Swift license](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/blob/main/LICENSE) · [Qwen license](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/blob/main/LICENSE-QWEN) · [Attribution notice](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/blob/main/NOTICE)

### Model Sources

- **Runtime and guide** — [antirez/ds4](https://github.com/antirez/ds4) · [Qwen3.8 Flash Next guide](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)
- **Upstream checkpoints** — [UkisAI Swift 1.5 weights](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next) · [UkisAI GGUF distribution](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF) · [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Demo** — UkisAI provides a side-by-side coding demo for the upstream models. [Swift 1.5 demo](https://ukisai.com/swift-games/flash-next)

## Uses

### Direct Use

Use the main GGUF with DS4 for local chat, summarization, coding assistance, and reasoning. Image input requires a compatible DS4 build and a matching vision projector.

### Downstream Use

Applications can use the model through `ds4-server`. Evaluate output quality, formatting, and tool-call reliability in the client and on the workloads you intend to run. [DS4 runtime documentation](https://github.com/antirez/ds4)

### Out-of-Scope Use

The model is unsuitable as the sole basis for high-stakes decisions or for unsupervised execution of actions with significant consequences.

## Bias, Risks, and Limitations

The model can produce incorrect facts, biased outputs, and faulty code. Shorter reasoning can trade off against accuracy or instruction following on some tasks. Quantization and runtime settings can affect output quality and latency. No demographic fairness assessment is available for this conversion.

### Recommendations

Review generated facts and code before relying on them. For deployments, test representative inputs across languages, context lengths, and modalities. Require human approval for consequential tool actions.

## How to Get Started with the Model

From the directory containing the GGUF, run the following with an existing DS4 build that supports the `qwen4exp` architecture. The example targets Apple Silicon. Start with a local SSD and a modest context allocation. [DS4 Qwen guide](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)

```sh
ds4 -m ./Swift-1.5-Qwen3.8-Flash-Next-DS4.gguf \
  --metal --ctx 8192 --prefill-chunk 1024 \
  -p "Explain how a hash table handles collisions."
```

Use the path to your existing `ds4` executable if it is outside your `PATH`. Use `--mtp` to try the embedded MTP layer. With compatible vision support, pass the matching projector as `--vision ./mmproj-Swift-1.5-Qwen3.8-Flash-Next-F16.gguf` and use `/read image.png` in interactive mode. Compatibility with the local projector files has yet to be verified for this card. [DS4 vision and MTP documentation](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)

## Training Details

### Training Data

UkisAI publishes a multi-turn agent dataset and reports resampling examples and adapting them into reinforcement learning environments. The DS4 conversion is a weight-packaging step. [Dataset card](https://huggingface.co/datasets/ukisai/Qwen3.8-27B-multi-turn-agent-sft) · [UkisAI training approach](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#training-approach)

### Training Procedure

UkisAI reports penalizing tokens associated with excessive reasoning, then using reinforcement learning and on-policy distillation to recover performance. The DS4 build uses a mix of MXFP4, Q8_0, and floating-point tensors. All three routed-expert projections in the main GGUF use MXFP4. The token embedding and output head are BF16. The conversion manifest records 128 n-gram shards. [UkisAI training approach](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#training-approach) · [Local conversion manifest](Swift-1.5-Qwen3.8-Flash-Next-DS4.gguf.json)

The Swift training precision, optimizer settings, training duration, and total compute are undisclosed in the upstream card. BF16 is the format of the checkpoint used in UkisAI’s published evaluation. [UkisAI evaluation](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#evaluation)

## Evaluation

### Testing Data, Factors, and Metrics

UkisAI compared its BF16 Swift checkpoint with BF16 Qwen3.8-Flash-Next at `xhigh` reasoning effort with MTP disabled. Its suite covers reasoning, math, coding, instruction following, and vision. It reports task scores and generated thinking-token counts. Terminal-Bench 2.1 uses total generated tokens. [UkisAI evaluation and settings](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#evaluation)

### Results

| Upstream BF16 benchmark | Qwen3.8-Flash-Next score | Swift 1.5 score | Change in mean generated tokens |
| --- | ---: | ---: | ---: |
| GPQA-Diamond | 89.80% | 89.60% | −55.8% |
| IFBench | 73.20% | 70.13% | −46.9% |
| LiveCodeBench v6 | 88.40% | 90.39% | −44.8% |
| Terminal-Bench 2.1 | 67.64% | 69.66% | +11.9% |

On GPQA-Diamond, UkisAI reports a 63.4% reduction in **median** thinking tokens. Results vary by task, including an increase in mean total output on Terminal-Bench 2.1. These figures are specific to UkisAI’s BF16 evaluation. [UkisAI results](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#evaluation)

A local DS4 evaluation log records an embedding retrieval stress test using this filename on an Apple M5 Max with 128 GiB RAM and vision disabled. In its small synthetic test, combined top-1 retrieval was 16.7% at an 8,192-token budget and 8.3% at 65,536 tokens, over 36 query-position observations per budget drawn from 12 queries. Long-input retrieval quality on representative data and generation quality or speed for this build remain unmeasured in the materials available for this card. Source: local DS4 checkout, `docs/EMBEDDING_EVALUATION.md`, recorded run dated 11 November 2026.

## Model Examination

The local GGUF header and sidecar manifest document the architecture, tensor types, and file layout. Interpretability results for the DS4 conversion are unavailable.

## Environmental Impact

The upstream card acknowledges compute support but gives no hardware inventory, training hours, compute region, electricity consumption, or CO₂e total. Energy use for the DS4 conversion and runtime has yet to be measured. [UkisAI acknowledgements](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#acknowledgements)

## Technical Specifications

### Model Architecture and Objective

Qwen3.8-Flash-Next combines Gated DeltaNet, Qwen Sparse Attention, gated residuals, and mixture-of-experts layers for autoregressive generation. Qwen describes 125B language-model parameters with 6B active, plus approximately 51B n-gram embedding parameters and 4B MTP parameters. The DS4 GGUF header reports 49 blocks including one MTP layer, 512 routed experts with 10 selected per token, and a native context length of 262,144 tokens. Usable context depends on the runtime and memory budget. [Qwen model overview](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [DS4 Qwen guide](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)

### Compute Infrastructure

The main file occupies **175,301,230,592 bytes (163.26 GiB)**. Its n-gram shards account for **95.37 GiB** according to the manifest. DS4 reads selected n-gram rows from disk. Keep the model on a fast local SSD and budget memory for the remaining weights, the context, and runtime buffers. The documented local test used an Apple M5 Max with 128 GiB RAM. Minimum supported memory for this conversion has yet to be established. [DS4 Qwen guide](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md) · [Local conversion manifest](Swift-1.5-Qwen3.8-Flash-Next-DS4.gguf.json)

| Local file or record | Role |
| --- | --- |
| `Swift-1.5-Qwen3.8-Flash-Next-DS4.gguf` | Main GGUF v3 file, architecture `qwen4exp` |
| `mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf` | Separate BF16 vision projector |
| `mmproj-Swift-1.5-Qwen3.8-Flash-Next-F16.gguf` | Separate FP16 vision projector |
| `swift-mtp-shared-Q8_0.gguf` | Separate Q8_0 MTP tail file. The main DS4 GGUF also has an embedded MTP layer. |
| `Swift-1.5-Qwen3.8-Flash-Next-DS4.gguf.json` | Conversion manifest with file size, source revision, and payload hashes |

The manifest records source revision `0bd4fe22431372cdad1979267d3ab45aa7e6150a` and SHA-256 `e90dbdfbeeb66901001715c2e512e985a0058290a578fff8ec5fc4f22cc75398`. The file size was checked locally. The hash is reproduced from the manifest. [Local conversion manifest](Swift-1.5-Qwen3.8-Flash-Next-DS4.gguf.json)

## Citation

Cite UkisAI for Swift 1.5 and credit Qwen for the base checkpoint and antirez for DwarfStar 4. [UkisAI citation](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF#citation) · [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) · [DwarfStar 4](https://github.com/antirez/ds4)

```bibtex
@misc{swift-qwen3.8-flash-next,
  title  = {Swift 1.5 Qwen3.8-Flash-Next},
  author = {{UkisAI}},
  year   = {2026},
  url    = {https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next}
}
```

UkisAI. (2026). *Swift 1.5 Qwen3.8-Flash-Next* [Large language model]. Hugging Face. [Model repository](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)

## Glossary

- **GGUF** — The binary format used here to store weights and model metadata.
- **MTP** — Multi-token prediction, used by DS4 for speculative decoding.
- **N-gram table** — Embeddings indexed by short token sequences. DS4 can read the required rows from disk.
- **MXFP4** — The block-scaled 4-bit floating-point format used for the routed expert weights in this build.

## Model Card Contact

A public contact for this local conversion is unconfirmed. For upstream checkpoint questions, use UkisAI’s model discussion tab. For DwarfStar 4 runtime issues, use the DS4 repository’s issue tracker. [UkisAI discussions](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/discussions) · [DwarfStar 4 issues](https://github.com/antirez/ds4/issues)
