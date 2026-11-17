---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
---

<!-- basalt-launch-buttons:start -->
<p align="center">
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-4bit&role=chat"><img src="https://img.shields.io/badge/%E2%96%B6_%D0%97%D0%B0%D0%BF%D1%83%D1%81%D1%82%D0%B8%D1%82%D1%8C_%D1%87%D0%B5%D1%80%D0%B5%D0%B7_Basalt-Apple_Silicon_MLX-0A84FF?style=for-the-badge&logo=apple&logoColor=white" alt="Запустить через Basalt"></a>
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-4bit&role=chat&lang=en"><img src="https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-macOS_App_%26_CLI-111111?style=for-the-badge&logo=swift&logoColor=F05138" alt="Run with Basalt"></a>
  <a href="https://bas-alt.vercel.app"><img src="https://img.shields.io/badge/Website-bas--alt.vercel.app-FF5E36?style=for-the-badge" alt="bas-alt.vercel.app"></a>
  <a href="https://github.com/goodmai/basalt/releases/tag/v0.11.0"><img src="https://img.shields.io/badge/Basalt-v0.11.0-0A84FF?style=for-the-badge&logo=github&logoColor=white" alt="Basalt v0.11.0"></a>
</p>

> **⚡ Запустить в [Basalt v0.11.0](https://github.com/goodmai/basalt/releases/tag/v0.11.0) · [bas-alt.vercel.app](https://bas-alt.vercel.app) (macOS 27+ · Swift 6.4 · Apple Silicon, без Python) · Upstream source: [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit):**
> ```bash
> basalt serve --model goodmai/Qwen3.8-27B-4bit --rest
> ```
<!-- basalt-launch-buttons:end -->

# mlx-community/Qwen3.8-27B-4bit

This model was converted to MLX format from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
using mlx-vlm version **0.6.8**.
Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for more details on the model.

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.8-27B-4bit --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
