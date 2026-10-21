---
pipeline_tag: any-to-any
base_model:
- google/gemma-4-26B-A4B-it
- google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
license: apache-2.0
license_name: apache-license-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: llama-cpp
tags:
- gemma4
- gemma4-26b
- a4b
- qat
- quantized
- gguf
- llama-cpp
- multimodal
language:
- multilingual
---

# Gemma 4 26B A4B IT QAT — GGUF

GGUF conversions derived from the Quantization-Aware Training (QAT) checkpoint of Gemma 4 26B A4B IT ([google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized)).

This repository provides multiple GGUF quantization levels, making the model usable with [llama.cpp](https://github.com/ggerganov/llama.cpp) and other GGUF-compatible inference engines.

## Model Overview

Gemma 4 26B A4B IT is a multimodal model built by Google DeepMind that handles **text**, **image**, and **audio** inputs and generates text output. It is designed for efficient on-device and server deployment.

## Model Architecture

| Property                      | Value                 |
| :---------------------------- | :-------------------- |
| **Architecture**              | Gemma4ForConditionalGeneration |
| **Parameters**               | 26B                   |
| **Supported Modalities**      | Text, Image, Audio    |
| **Context Length**            | 256K tokens (262144)  |
| **Vocabulary Size**           | 262K (262144)         |

## GGUF Files

| File                                | Description                                      |
| :---------------------------------- | :----------------------------------------------- |
| `gemma-4-26b-a4b-qat-Q4_0.gguf`     | Q4_0 Quantization                                |
| `gemma-4-26b-a4b-qat-Q4_K_M.gguf`   | Q4_K_M Quantization                              |
| `gemma-4-26b-a4b-qat-Q4_K_S.gguf`   | Q4_K_S Quantization                              |
| `gemma-4-26b-a4b-qat-Q5_K_M.gguf`   | Q5_K_M Quantization                              |
| `gemma-4-26b-a4b-qat-Q5_K_S.gguf`   | Q5_K_S Quantization                              |
| `gemma-4-26b-a4b-qat-Q6_K.gguf`     | Q6_K Quantization                                |
| `gemma-4-26b-a4b-qat-Q8_0.gguf`     | Q8_0 Quantization                                |
| `gemma-4-26b-a4b-qat.gguf`         | Default/Full QAT GGUF                             |
| `mmproj.gguf`                       | Multimodal projector (vision + audio)            |

A `chat_template.jinja` file is also provided for use with chat-based inference.

## Usage

### llama.cpp (CLI)

```bash
# Run text-only inference
./llama-cli \
  -m gemma-4-26b-a4b-qat-Q4_K_M.gguf \
  -p "Explain quantum computing in simple terms." \
  --temp 1.0 --top-k 64 --top-p 0.95
```

### llama-server (OpenAI-compatible API)

```bash
# Text-only
./llama-server \
  -m gemma-4-26b-a4b-qat-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080

# Multimodal (image + audio)
./llama-server \
  -m gemma-4-26b-a4b-qat-Q4_K_M.gguf \
  --mmproj mmproj.gguf \
  --host 0.0.0.0 --port 8080
```

### Multimodal (Image / Audio)

For image and audio inputs, use `llama-server` or `llama-cli` with the `--mmproj` flag pointing to `mmproj.gguf`. Refer to your inference engine's documentation for passing image/audio data alongside text prompts.

> **Modality order tip:** For best results, place image content **before** text and audio content **after** text in your prompt.

## Generation Parameters

Recommended parameters:

| Parameter       | Value  |
| :-------------- | :----- |
| Temperature     | 1.0    |
| Top-K           | 64     |
| Top-P           | 0.95   |
| BOS Token ID    | 2      |
| EOS Token ID    | 1      |
| Pad Token ID    | 0      |
| Mask Token ID   | 4      |

## Thinking Mode

Gemma 4 supports configurable thinking (reasoning) mode:

- **Enable:** Include the `<|think|>` token at the start of the system prompt.
- **Output format:** When thinking is enabled, the model outputs internal reasoning followed by the final answer:
  ```
  <|channel>thought
  [Internal reasoning]
  <channel|>
  [Final answer]
  ```
- **Disable:** Omit the `<|think|>` token.

## Key Features

- **Multimodal:** Text, image, and audio understanding
- **Long Context:** 256K token context window
- **Function Calling:** Native support for structured tool use (agentic workflows)
- **Multilingual:** Support for 140+ languages
- **Native System Prompt:** Supports the `system` role for structured conversations

## Acknowledgements

- Original model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- QAT checkpoint: [google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized)
- Technical report: [Gemma 4 Technical Report (arXiv:2607.02770)](https://arxiv.org/abs/2607.02770)

## Citation

```bibtex
@misc{gemmateam2026gemma4,
      title={Gemma 4 Technical Report},
      author={Gemma Team},
      year={2026},
      eprint={2607.02770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.02770},
}
```

## License

[Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license)
