---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- nvfp4
- vllm
- quantized
---

# Qwen3.8-27B, NVFP4

Qwen3.8-27B in NVFP4, 4-bit, for NVIDIA Blackwell GPUs. Kapitsa does not run this build.

| For a Mac, the Kapitsa builds | Format |
| --- | --- |
| [nace-ai/Qwen3.8-27B-MLX-3bit](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-3bit) | MLX |
| [nace-ai/Qwen3.8-27B-MLX-4bit](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-4bit) | MLX |
| [nace-ai/Qwen3.8-27B-MLX-8bit](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-8bit) | MLX |
| [nace-ai/Qwen3.8-27B-GGUF](https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF) | GGUF |

## Run it

```bash
pip install vllm
vllm serve nace-ai/Qwen3.8-27B-NVFP4
```

| Engine | Status |
| --- | --- |
| vLLM | tested with vLLM 0.33.0 on a B200 (Modal), one serve + generations (GSM8K-100 + speed runs) |
| TensorRT-LLM | not tried |

## Fit and build

| Download | Made with | Tokenizer |
| ---: | --- | --- |
| 20.6 GB | llm-compressor (compressed-tensors) for vLLM / TensorRT-LLM on Blackwell | byte-for-byte the one in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |

| Test | This variant | Reference | Verdict |
| --- | ---: | ---: | --- |
| GSM8K (100 questions) | 0.950 (n=100) | bf16 0.940 (n=100) | matches |

| Known warning |
| --- |
| vLLM warns the checkpoint's NVFP4 global scale differs across parallel (q/k/v) layers ('reduced accuracy possible'); measured GSM8K-100 0.95 vs 0.94 bf16 shows no practical hit. |

## Speed

| GPU | Engine | Writes (tok/s) | Reads a 4K prompt (tok/s) | First token (ms) | Peak memory (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| NVIDIA B200 | vLLM 0.33.0 | 195 | 10,781 | 378 | 150.8 |

| How measured | |
| --- | --- |
| Run | single stream, a prompt of about 4K tokens, 512 tokens written greedily |
| Peak memory | the GPU memory the server held, which includes memory the engine reserves up front, so it is not the model's size |

## License

apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
