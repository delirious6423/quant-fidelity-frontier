---
base_model:
- Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
---
Bit-for-bit mirror of Qwen/Qwen3.8-27B-FP8;

only text_config.rope_parameters changed to YaRN factor 2.0 (workaround for vLLM #58080 MTP-draft context);

base model Qwen/Qwen3.8-27B, Apache-2.0