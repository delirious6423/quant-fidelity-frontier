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

# Qwen3.8-27B-Q5-GGUF

A 5-bit-class XQD quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in GGUF.

Made with post-training quantization only, using our own calibration dataset and importance matrix. No fine-tuning. On our held-out agentic set it has 23.5% lower KLD than kaitchup's SLQ-EAR98 at the same size. How it is made and measured is in our write-up, [Old ideas, standard formats, better quants](https://axolotlai.substack.com/p/old-ideas-standard-formats-better).

| file | size |
|---|---:|
| `Qwen3.8-27B-XQD-Q5.gguf` | 18.85 GB |

Text only: no vision tensors or multimodal projector are included. The file keeps the model's multi-token-prediction (MTP) block.

## Usage

Use a recent llama.cpp build with Qwen3.8 support.

```bash
hf download axolotl-ai-co/Qwen3.8-27B-Q5-GGUF Qwen3.8-27B-XQD-Q5.gguf --local-dir .
llama-server --model Qwen3.8-27B-XQD-Q5.gguf --n-gpu-layers 99 --ctx-size 32768
```

## License

Apache 2.0, inherited from the base model. See [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for the original license and terms.
