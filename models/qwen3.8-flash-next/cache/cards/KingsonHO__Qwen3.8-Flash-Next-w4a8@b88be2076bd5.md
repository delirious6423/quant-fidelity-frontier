---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
frameworks: PyTorch
base_model_relation: quantized
base_model:
  - Qwen/Qwen3.8-Flash-Next
tasks:
  - text-generation
---
# Qwen3.8-Flash-Next-w4a8

## 1. 基本信息

| 项目 | 信息 |
| --- | --- |
| 原始模型名 | Qwen3.8-Flash-Next |
| 原始模型链接 | [Qwen3.8-Flash-Next](https://www.modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) |
| 量化工具 | ModelSlim 26.1.0 |
| msmodelslim commit id | `cfb2c0077ec78678983830fab096fbed4176b01e`，含本地 Flash-Next 适配 |
| 精度测试机型 | Ascend 910，8 卡 |
| 精度测试平台 | docker vllm-ascend |
| 版本 | `vllm-ascend:v0.32.0rc-a3-openeuler-cann9.2.1-20261102-r67` |

## 2. 量化脚本

```python3
python3 -m msmodelslim.cli quant \
  --model_path {浮点权重路径} \
  --save_path {W4A8量化权重路径} \
  --device npu \
  --model_type Qwen3.8-Flash-Next \
  --config {量化配置路径} \
  --trust_remote_code true
```

## 3. 精度测试结果

| 模型名 | 量化格式 | 数据集 | 测试精度 % | 官方精度 % |
| --- | --- | --- | ---: | ---: |
| Qwen3.8-Flash-Next-w4a8 | INT W4A8 / W8A8 混合精度 | GPQA Diamond | 90.91（180/198） | 91.7 |

使用 AIS Bench，0-shot CoT，输出上限 65,536 tokens，batch size=32；temperature=1.0、top_p=0.95、top_k=20。MTP=3，async 开启
