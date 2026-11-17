---
model_name: Qwen3.8-27B 3-bit MLX Text-Only
library_name: mlx
pipeline_tag: text-generation
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - mlx-lm
  - safetensors
  - qwen
  - qwen3.8
  - qwen3.8-27b
  - qwen3_5
  - text-generation
  - text-only
  - conversational
  - reasoning
  - tool-use
  - long-context
  - multilingual
  - quantized
  - weight-only-quantization
  - affine
  - rtn
  - group-size-64
  - 3-bit
---

<!-- basalt-launch-buttons:start -->
<p align="center">
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-3bit-MLX-TextOnly&role=chat"><img src="https://img.shields.io/badge/%E2%96%B6_%D0%97%D0%B0%D0%BF%D1%83%D1%81%D1%82%D0%B8%D1%82%D1%8C_%D1%87%D0%B5%D1%80%D0%B5%D0%B7_Basalt-Apple_Silicon_MLX-0A84FF?style=for-the-badge&logo=apple&logoColor=white" alt="Запустить через Basalt"></a>
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-3bit-MLX-TextOnly&role=chat&lang=en"><img src="https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-macOS_App_%26_CLI-111111?style=for-the-badge&logo=swift&logoColor=F05138" alt="Run with Basalt"></a>
  <a href="https://github.com/goodmai/basalt/releases/tag/v0.11.0"><img src="https://img.shields.io/badge/Basalt-v0.11.0-0A84FF?style=for-the-badge&logo=github&logoColor=white" alt="Basalt v0.11.0"></a>
</p>

> **⚡ Запустить в [Basalt v0.11.0](https://github.com/goodmai/basalt/releases/tag/v0.11.0) (macOS 27+ · Swift 6.4 · Apple Silicon, без Python) · Upstream source: [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly):**
> ```bash
> basalt serve --model goodmai/Qwen3.8-27B-3bit-MLX-TextOnly --rest
> ```
<!-- basalt-launch-buttons:end -->

# lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly

This quantization strips the model's vision tower and preserves only text-to-text capabilities. For a vision capable version check [lukaskremla/Qwen3.8-27B-3bit-MLX](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX).

You can download MTP heads separately from this collection [Qwen 3.8 27B MLX-Quants (Vision, Text-Only & MTP)](https://huggingface.co/collections/lukaskremla/qwen-38-27b-mlx-quants-vision-text-only-and-mtp).

Hugging Face might render incorrect parameter counts for this model, it is a common display bug for MLX quants.

## Credits

Converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using mlx-lm version **0.31.2**.
