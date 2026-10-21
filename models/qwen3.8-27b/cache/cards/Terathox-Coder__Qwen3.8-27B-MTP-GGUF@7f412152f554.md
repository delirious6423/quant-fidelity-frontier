---
license: apache-2.0
language:
  - en
  - es
  - zh
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - gguf
  - qwen
  - qwen3.8
  - 27b
  - q4_k_m
  - mtp
  - nextn
  - llama.cpp
  - ollama
  - quantized
  - conversational
---

# Qwen3.8-27B MTP GGUF — Q4_K_M

Community **GGUF Q4_K_M quantization of `Qwen/Qwen3.8-27B` with the original MTP / NextN tensors preserved**.

This repository contains a format conversion and quantization of the original Qwen3.8-27B checkpoint.

**No fine-tuning or additional training has been performed.**

## Model Details

| Property     | Value              |
| ------------ | ------------------ |
| Base model   | `Qwen/Qwen3.8-27B` |
| Parameters   | 27B                |
| Quantization | `Q4_K_M`           |
| Format       | GGUF               |
| MTP / NextN  | Preserved          |
| File size    | ~16.8 GB           |
| Conversion   | `llama.cpp`        |
| License      | Apache-2.0         |

### Available File

```text
Qwen3.8-27B-Q4_K_M-MTP.gguf
```

## MTP / NextN

The original Qwen3.8-27B checkpoint contains Multi-Token Prediction / NextN tensors.

During conversion:

```text
MTP enabled: 866 tensors
NextN disabled: 851 tensors
Difference: 15 MTP tensors
```

The 15 additional `mtp.*` tensors were intentionally preserved in this GGUF.

The current `llama.cpp` converter recognized the MTP export path and correctly mapped the NextN tensors into the additional model block.

## llama.cpp Validation

The resulting Q4_K_M GGUF was successfully loaded and executed with `llama.cpp` using:

```bash
llama-cli \
  -m Qwen3.8-27B-Q4_K_M-MTP.gguf \
  --spec-type draft-mtp
```

The runtime successfully:

* loaded the GGUF;
* recognized the model architecture;
* recognized the embedded MTP / NextN tensors;
* constructed the main and MTP graphs;
* processed the prompt;
* generated output through the MTP-compatible runtime path.

## Ollama

The GGUF has also been successfully imported and executed with **Ollama 0.32.9**.

Minimal `Modelfile`:

```text
FROM ./Qwen3.8-27B-Q4_K_M-MTP.gguf
```

Create:

```bash
ollama create qwen3.8:27b-mtp-q4_K_M -f Modelfile
```

Run:

```bash
ollama run qwen3.8:27b-mtp-q4_K_M
```

### MTP Configuration

For runtimes that expose MTP speculative decoding, the local TERATHOX configuration uses:

```text
draft_num_predict = 4
```

Note that **loading a GGUF containing MTP tensors does not by itself guarantee that a runtime is actively using speculative MTP decoding**.

Users should verify MTP support and configuration for their specific runtime version.

## Local TERATHOX Deployment

This quantization has been tested locally under the alias:

```text
Terathox-Coder:Nova
```

### Hardware

```text
NVIDIA GeForce RTX 5080  16 GB
NVIDIA GeForce RTX 4070  12 GB
NVIDIA GeForce RTX 4070  12 GB
```

Three GPUs were used for the local validation.

### Ollama Runtime Configuration

```text
Context:                  204800
OLLAMA_FLASH_ATTENTION:   1
OLLAMA_VULKAN:            false
OLLAMA_KV_CACHE_TYPE:     q4_0
OLLAMA_SCHED_SPREAD:      false
OLLAMA_GPU_OVERHEAD:      0
OLLAMA_NUM_PARALLEL:      1
OLLAMA_MAX_LOADED_MODELS: 1
OLLAMA_KEEP_ALIVE:        -1
draft_num_predict:        4
```

Observed status:

```text
NAME                   SIZE     PROCESSOR    CONTEXT
Terathox-Coder:Nova    24 GB    100% GPU     204800
```

## Local Performance

Observed interactive generation performance:

| Run |   Eval rate |
| --- | ----------: |
| 1   | 49.90 tok/s |
| 2   | 49.52 tok/s |
| 3   | 54.18 tok/s |
| 4   | 49.32 tok/s |

Typical observed generation range:

```text
~49–54 tokens/s
```

Prompt evaluation varied depending on conversation state and cached context, reaching values from approximately:

```text
51 tok/s → 263 tok/s
```

These are **local hardware measurements and not standardized model benchmarks**.

Performance depends on hardware, context size, GPU offload, KV-cache configuration, runtime version and MTP implementation.

## Intended Use

This GGUF is intended for:

* local text generation;
* coding and software engineering;
* agentic coding workflows;
* technical reasoning;
* long-context workloads;
* experimentation with MTP / NextN speculative decoding;
* local inference with `llama.cpp` or compatible GGUF runtimes.

## Limitations

This is a quantized derivative of the original model.

`Q4_K_M` significantly reduces memory requirements but may introduce some quality degradation compared with the original BF16 checkpoint.

The base model may also produce inaccurate, biased or hallucinated information. Outputs should be independently verified for high-impact or safety-critical use cases.

## Vision / Multimodal Support

The original Qwen3.8-27B model includes multimodal capabilities.

**This repository currently provides the GGUF language-model artifact only.**

No independently validated multimodal projector (`mmproj`) is currently included in this repository.

Therefore this release should currently be considered **text-oriented unless an appropriate multimodal projector is added and validated**.

## Training

No training or fine-tuning was performed for this repository.

The original weights come from:

```text
Qwen/Qwen3.8-27B
```

This repository only performs:

```text
Original checkpoint
        ↓
GGUF BF16 with MTP preserved
        ↓
Q4_K_M quantization
        ↓
Qwen3.8-27B-Q4_K_M-MTP.gguf
```

## Datasets

No additional dataset was used.

This repository does not contain a fine-tuned model.

## Evaluation

No standardized quality benchmark was performed specifically on this quantization at the time of publication.

The performance results above measure **local inference throughput only** and should not be interpreted as accuracy or capability benchmarks.

For official capability benchmarks, refer to the original `Qwen/Qwen3.8-27B` model card.

## Attribution

Original foundation model developed by the **Qwen Team**.

Base model:

```text
Qwen/Qwen3.8-27B
```

GGUF conversion and Q4_K_M quantization:

```text
Terathox-Coder
```

The original MTP / NextN tensors were preserved during conversion.

**TERATHOX does not claim authorship or training of the original Qwen foundation model.**

## License

This repository follows the **Apache License 2.0** of the base model.

Please review the original Qwen3.8-27B repository and license for additional information.

## Disclaimer

This is a **community conversion** and is not an official Qwen release.

Compatibility, performance and MTP behavior may vary between versions of `llama.cpp`, Ollama and other GGUF runtimes.
