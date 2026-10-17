---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
- qwen
- qwen3.8
- awq
- int4
- quantized
license: apache-2.0
---

# Kamnsv/Qwen3.8-27B-AWQ-INT4

AWQ INT4-квантованная версия [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).

Квантование выполнено локально. Модель предназначена для инференса через совместимые AWQ-бэкенды, например vLLM.

## Использование

```bash
vllm serve Kamnsv/Qwen3.8-27B-AWQ-INT4 \
  --quantization awq \
  --max-model-len 32768
```

## Замечания

- Оригинальная модель: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
- Формат весов: AWQ INT4.
- Проверьте совместимость версии vLLM с архитектурой `Qwen3.8`.
