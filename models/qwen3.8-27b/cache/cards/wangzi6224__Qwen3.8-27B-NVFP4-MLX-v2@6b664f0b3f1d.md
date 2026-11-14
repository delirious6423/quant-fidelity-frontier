---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
library_name: mlx
---

# Qwen3.8-27B NVFP4（MLX 格式）

由 Ollama 模型库的 `qwen3.8:27b-mlx` 转换而来（清单摘要 `sha256:5642e97495e1a088883805981563dcdc4a040c2f53388b7a41d1f24d3622cf7e`，2026-09-22 拉取；同一份内容在标签页上也叫 `27b-nvfp4`），基础模型是 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。只含语言部分，约 17GB。

## 来源说明

- 原包附带的许可是 Apache License 2.0 全文（见 `LICENSE`）。
- 原包的量化说明（`hf_quant_config.json`）写的工具是 NVIDIA ModelOpt（版本号 0.0.0），`MIXED_PRECISION`：400 个线性层是 NVFP4（16 个一组），另有 9 个只量化权重（`W4A16_NVFP4`，含输出层）。原包没有写明量化的执行者。
- 不量化的张量（层归一化等）和 Qwen 官方 `Qwen/Qwen3.8-27B`（revision `1d4bf0f2`）的差约 4–6%，和 `Inferact/Qwen3.8-27B-NVFP4` 的逐字节相同，底子不是官方公开的那份权重。

## 改动（Apache-2.0 第 4 条要求注明）

- 张量只改键名、不改字节：`model.language_model.*` 改成 `language_model.model.*`，`lm_head.*` 改成 `language_model.lm_head.*`，`.weight.scale` 改成 `.scales`。
- 每个量化层的 `weight.global_scale`（原值是幅值除以 448×6）乘 448×6，放在 `global_scales.safetensors`，键名 `<层>.global_scale`。加载时需要把这里的值设到对应量化层的 `global_scale` 上；只读主权重文件的加载器会得到错误结果。
- 去掉视觉部分和 MTP 头。
- `config.json` 的量化字段改成 `{"mode": "nvfp4", "group_size": 16, "bits": 4}`；`config.json`、`chat_template.jinja`、`generation_config.json`、`merges.txt` 取自 `wangzi6224/Qwen3.8-27B-NVFP4-MLX`（revision `80fd7213`）；三个分词器文件和原包里的逐字节相同。
- `SHA256SUMS` 列出全部文件的校验值。

## 许可

Apache License 2.0，见 `LICENSE`（取自原包）。

---

MLX-format conversion of the `qwen3.8:27b-mlx` package (manifest `sha256:5642e97495e1a088883805981563dcdc4a040c2f53388b7a41d1f24d3622cf7e`, pulled 2026-09-22), based on Qwen/Qwen3.8-27B. Language model only (~17 GB).
The source package ships the Apache License 2.0 and states NVIDIA ModelOpt `MIXED_PRECISION` NVFP4 quantization; who ran the quantization is not stated. Its unquantized tensors differ from the official Qwen/Qwen3.8-27B release by about 4–6% and are byte-identical to Inferact/Qwen3.8-27B-NVFP4.
Changes: tensors renamed only (bytes unchanged); per-layer global scales multiplied by 448×6 and stored in `global_scales.safetensors` (the loader must apply them); vision tower and MTP head removed; config, chat template and generation config taken from wangzi6224/Qwen3.8-27B-NVFP4-MLX@80fd7213. License: Apache-2.0 (see LICENSE).
