---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- qwen3_8
- moe
- fp8
- reproducibility
---

# Qwen3.8-Flash-Next-FP8-Recreated

An independent recreation of [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
from the BF16 release [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (rev `de4b8e4d`):
**all 152,089 tensors are bitwise identical to the official FP8 checkpoint** (rev `236dfdf2`). For serving, use the
official repo; this one documents the quantization recipe. Not affiliated with or endorsed by Qwen.

**Full model card (recipe and verification details) coming soon.**

## License

Qwen Community License 1.0, Copyright (c) 2026 Qwen ([LICENSE](LICENSE)). Changes made here (TK-13, 2026-10-24):
`model-*.safetensors` (BF16 release quantized to FP8), `config.json` (`quantization_config` added, identical to the
official FP8 config), `README.md` replaced. All other files are unmodified from Qwen/Qwen3.8-Flash-Next.
