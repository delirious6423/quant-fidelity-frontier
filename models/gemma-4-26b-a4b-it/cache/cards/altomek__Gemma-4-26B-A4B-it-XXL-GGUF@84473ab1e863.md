---
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
---

# Gemma-4-26B-A4B-it

GGUF quant of https://huggingface.co/google/gemma-4-26B-A4B-it

## Changelog

- reupload 1: better recipe to quantize weights = smaller quant & better performance + BOS added
- reupload 2: updated chat_template as in: https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/21
- reupload 3: added updated tokenizer_config.json