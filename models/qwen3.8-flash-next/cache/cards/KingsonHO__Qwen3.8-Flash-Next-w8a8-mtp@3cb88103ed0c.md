---
frameworks:
  - PyTorch
license: other
license_name: qwen-community-1.0
license_link: https://www.modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/file/view/master/LICENSE
base_model_relation: quantized
base_model:
  - Qwen/Qwen3.8-Flash-Next
tasks:
  - image-text-to-text
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next-w8a8-mtp

## 1. 基本信息

| 项目 | 信息 |
|:------:|:------:|
| 原始模型名 | Qwen3.8-Flash-Next |
| 原始模型链接 | [Qwen/Qwen3.8-Flash-Next](https://www.modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) |
| msmodelslim私仓及分支 | https://gitcode.com/code_mingming/msmodelslim/tree/master-qwen38-1022 |
| msmodelslim commit id | cfb2c0077ec78678983830fab096fbed4176b01e |

## 2. 量化脚本

```bash
msmodelslim quant \
    --model_path weights/Qwen3.8-Flash-Next \
    --save_path weights/Qwen3.8-Flash-Next-w8a8-mtp \
    --device npu \
    --device_id 0 1 2 3 4 5 6 7 \
    --model_type Qwen3.8-Flash-Next \
    --config .yaml \
    --trust_remote_code true
```

## 3. 精度测试结果

| 模型名 | 量化格式 | 数据集 | 测试精度 % | 官方精度 % |
|:------:|:------:|:------:|:------:|:------:|
| Qwen3.8-Flash-Next-w8a8-mtp | W8A8 | GPQA-Diamond | 91.41 | 91.7 |
