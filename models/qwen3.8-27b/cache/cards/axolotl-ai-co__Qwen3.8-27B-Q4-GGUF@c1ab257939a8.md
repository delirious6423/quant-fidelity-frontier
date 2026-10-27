---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- llama.cpp
- quantized
---

# Qwen3.8-27B-Q4-GGUF

A 4-bit-class GGUF quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

Made with post-training quantization only, using our own calibration dataset and importance matrix. No fine-tuning.

| file | size |
|---|---:|
| `qwen38-q4kxl-alloc.gguf` | 17.55 GB |

Text only: no vision tensors or multimodal projector are included. The file keeps the model's multi-token-prediction (MTP) block.

## Usage

Use a recent llama.cpp build with Qwen3.8 support.

```bash
hf download axolotl-ai-co/Qwen3.8-27B-Q4-GGUF qwen38-q4kxl-alloc.gguf --local-dir .
llama-server --model qwen38-q4kxl-alloc.gguf --n-gpu-layers 99 --ctx-size 32768
```

## License

Apache 2.0, inherited from the base model. See [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for the original license and terms.
