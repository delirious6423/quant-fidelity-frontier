---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.8
  - multimodal
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B (GGUF)

Q4_K_M GGUF build of **Qwen3.8-27B**, packaged for local use in PrivaEmanator.

## Source

This model is **not** trained, modified, or quantized by us. All credit goes to the original authors:

- **GGUF quantization:** [lmstudio-community/Qwen3.8-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF) by **LM Studio Community**
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the **Qwen team**

This repo only redistributes those files in a layout PrivaEmanator can load directly.

## Files

| File | Description |
| --- | --- |
| `Qwen3.8-27B-Q4_K_M.gguf` | Language model, Q4_K_M quantization (~16.8 GB) |
| `mmproj-BF16.gguf` | Vision projector for image input |
| `chat_template.jinja` | Chat template, thinking disabled |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | Tokenizer |
| `config.json`, `generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json` | Model, generation and image/video preprocessing config |

## Usage

Download the whole repo into one folder (e.g. `assets/models/qwen3.8`) and select the `.gguf` model in PrivaEmanator. The `mmproj` file in the same folder enables image input.

## License

Apache-2.0, following the base model. See `LICENSE`.
