---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- deepseek-v4
- gguf
- llama.cpp
- reap
- quantized
---

# DeepSeek-V4-Flash-Spark-Mini GGUF (Q3-Dynamic REAP)

GGUF quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT),
Mini variant of the REAP-pruned Spark lineage with dynamic per-layer Q3 allocation.

## Files

- `DeepSeek-V4-Flash-Spark-Mini-Q3-Dynamic-REAP-ds4.gguf` — single-file GGUF
- `DeepSeek-V4-Flash-Spark-Mini-Q3-Dynamic-REAP-ds4.gguf.sha256` — checksum sidecar

## Usage

Compatible with llama.cpp and other GGUF loaders:

```bash
llama-server -m DeepSeek-V4-Flash-Spark-Mini-Q3-Dynamic-REAP-ds4.gguf
```

## License

MIT, inherited from the upstream DeepSeek-V4-Flash-0731 repository.
See the upstream model card for full terms.

## Acknowledgements

- [DeepSeek](https://huggingface.co/deepseek-ai) for the upstream [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT) this quantization derives from.
- The llama.cpp project and community for the GGUF format and runtime support.

