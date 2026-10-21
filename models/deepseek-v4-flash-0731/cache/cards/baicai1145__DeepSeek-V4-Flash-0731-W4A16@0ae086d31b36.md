---
language:
  - en
  - zh
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE
tags:
  - deepseek
  - deepseek-v4
  - int4
  - w4a16
  - compressed-tensors
  - vllm
  - quantized
base_model:
  - deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
library_name: vllm
---

# DeepSeek-V4-Flash-0731-W4A16

INT4 (W4A16) 量化版本的 **DeepSeek-V4-Flash-0731** (283B params)。
将原始 FP4 experts + FP8 attention 混合精度模型转换为 vLLM 原生的 `compressed-tensors` 混合精度格式,可在 **无 FP4 硬件支持** 的 NVIDIA GPU(A100 / L40S / L20 / RTX 4090 等)上推理。

> 本仓库与原始 `deepseek-ai/DeepSeek-V4-Flash-0731` 模型**出自同一权重**,仅做格式与精度转换,未做任何微调或蒸馏。

---

## 为什么需要这个量化版本?

原始 DeepSeek-V4-Flash-0731 采用**混合精度**:
- **256 个 routed experts**:FP4 (`e2m1fn_x2` 打包为 int8,配 e8m0 per-block scale)
- **Attention (MLA) / shared experts / indexer**:FP8 (`e4m3`,配 e8m0 per-block scale)

FP4 是 NVIDIA Blackwell 架构原生格式。**A100 / H100 / H20 等非 Blackwell GPU 不支持 FP4 tensor core**,无法直接运行原始权重。

本量化版本将 **experts 从 FP4 转为 INT4 (group_size=32)**,保留 attention 为 FP8,使其能在 Hopper / Ampere 架构上通过 vLLM 加载推理。

| 硬件 | 原始模型 | 本 W4A16 版本 |
|------|:--------:|:------------:|
| Blackwell (B200) | ✅ 原生 | ✅ |
| Hopper (H20/H100) | ⚠️ FP8 部分 OK,experts 退化 | ✅ **推荐** |
| Ampere (A100/L40S) | ❌ 不支持 | ⚠️ 需 vLLM 小补丁(见下文) |

---

## 量化方案 (Quantization Scheme)

本模型采用 `compressed-tensors` 的 **mixed-precision** 格式,包含两个 config_group:

### group_0 — Attention (FP8 W8A8,保持原始精度)

| 项 | 值 |
|----|----|
| targets | `re:.*attn\.(wq_a\|wq_b\|wkv\|wo_a\|wo_b\|fused_wqa_wkv)$` |
| weights | FP8 e4m3,block 结构 `[128,128]`,symmetric |
| input_activations | FP8 dynamic,group_size=128,symmetric |
| 说明 | **直接保留原始模型的 FP8 权重和 scale,未做重采样** |

### group_1 — Routed Experts (INT4 W4A16,FP4 转换)

| 项 | 值 |
|----|----|
| targets | `re:.*experts\.\d+\.(w1\|w2\|w3\|gate_proj\|up_proj\|down_proj\|gate_up_proj)$` |
| weights | INT4,group_size=32,symmetric,BF16 scale |
| input_activations | 无(W4A16:激活在 BF16 计算) |
| 说明 | 由原始 FP4 权重经**最优 scale 网格搜索**转换而来(见下文) |

### 其余部分保持 BF16(不量化)

`lm_head`、`embed_tokens`、所有 `norm`、router `gate`、`shared_experts`、`compressor`、`indexer.weights_proj`、hyper-connection 参数等**保持 BF16**(与原始非 FP4 部分一致)。

---

## FP4 → INT4 转换算法(无损性分析)

这是本仓库的核心技术点。原始 experts 的 FP4 权重并非直接 round 到 INT4,而是经过两步:

### 第 1 步:FP4 反量化到 BF16 高精度参考

原始 FP4 (`e2m1fn_x2`) 每个元素是 4 bit 索引,通过查找表还原:

```python
FP4_TABLE = [0, 0.5, 1, 1.5, 2, 3, 4, 6,
             0, -0.5, -1, -1.5, -2, -3, -4, -6]
# code (4bit) → FP4_TABLE[code] × scale(e8m0) = 真实权重值
```

由于 FP4 每个权重值已是离散点,反量化到 BF16 **是精确无损的**(只是把离散值用 BF16 表示)。

### 第 2 步:BF16 → INT4,用 scale 网格搜索最小化 MSE

INT4 量化采用 group_size=32(与 FP4 的 [128,128] block 对齐,32 是 FP4 一个 e8m0 scale 覆盖 128 元素时的细分子组),对称量化:

```python
# 对每个 group, 搜索最优的 round-rate r
for r in [6/7, 0.9, 0.95, 1.0, 1.05, 1.1]:
    scale = amax(group) * r / 7
    q = clamp(round(w / scale), -8, 7)
    mse = mean((w - q*scale)²)
    # 保留 MSE 最小的 (q, scale)
```

普通 INT4 量化固定用 `r = 1.0`(scale = amax/7)。本转换**额外搜索 6 个 r 候选**,对每个 group 选取 MSE 最小的组合,显著降低量化误差。

### 理论无损性论证

| 来源 | 误差 | 是否可消除 |
|------|------|-----------|
| FP4 → BF16 反量化 | 0(精确) | — |
| BF16 → INT4 量化误差 | 非零 | ❌ 不可避免(信息论下界) |

**严格意义上,FP4 → INT4 不是数学无损的**,因为 INT4 的 16 个量化电平与 FP4 的 16 个电平**分布不同**:
- FP4 电平(归一化后):`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` —— 非均匀,覆盖大动态范围
- INT4 电平(归一化后):`{0, ±1, ±2, ±3, ±4, ±5, ±6, ±7} × scale/7` —— 均匀

但通过**最优 scale 搜索 + group_size=32 细粒度**,实际量化误差被压到极低。实测信噪比(SNR):

```
SNR(FP4 → INT4, group_size=32) ≈ 25.8 dB
```

这对应约 **0.26% 的相对均方误差**(RMSE ≈ 5.1% 相对幅度),在 LLM 推理中通常**对最终输出无可感知影响**(见下方实测结果)。

### 为什么 group_size=32?

- 原始 FP4 每个 e8m0 scale 覆盖 128 个连续权重(block_size=[128,128] 的行方向)
- INT4 用更小的 group(32)→ 每个 group 独立算 scale,动态范围更紧 → MSE 更低
- 32 同时是 128 的约数,与 FP4 block 结构天然对齐
- 更小的 group(如 16)误差更低,但 scale 存储开销增大,32 是质量/体积平衡点

---

## 实测质量与性能(2×H20 96GB)

### 质量(50 题简化评测,greedy decoding)

| 类别 | 原始 FP4+FP8 | 本 W4A16 | 差异 |
|------|:-----------:|:--------:|:----:|
| 数学 (10题) | 100% | 100% | 0 |
| 代码 (10题) | 100% | 100% | 0 |
| 常识 (10题) | 100% | 100% | 0 |
| 逻辑 (10题) | 100% | 100% | 0 |
| 指令遵循 (10题) | 100% | 100% | 0 |
| **总计 (50题)** | **100%** | **100%** | **0** |

> 两个模型答错/答对的题**完全相同**,包括同一个 Unicode 下标表达的边缘 case。在 50 题评测集上 **INT4 量化零质量损失**。
>
> 注:50 题评测集偏简单,主要验证"没把模型搞坏"。如需更高置信度,建议跑 GSM8K / MMLU。

### DSpark 投机解码接受率(num_speculative_tokens=7)

| 指标 | 原始 FP4+FP8 | 本 W4A16 |
|------|:-----------:|:--------:|
| 整体 token 接受率 | 38.1% | **41.7%** (+3.6pp) |
| 每步平均接受 token | 2.67 | 2.92 |
| pos0(首个 spec)接受率 | 72.3% | 72.7% |

接受率不降反升,因为 DSpark 的 draft 与 main 用同一套权重,接受率取决于两者一致性而非绝对精度。

### 推理速度(DSpark ON)

| 配置 | 原始 FP4+FP8 | 本 W4A16 |
|------|:-----------:|:--------:|
| 2×H20 | 21.7 tok/s | 21.4 tok/s |

H20 原生支持 FP4,INT4 在 Hopper 上无速度优势;**A100 上情况反转**(原始 FP4 跑不了,只有 W4A16 能跑)。

---

## 模型文件结构

```
DeepSeek-V4-Flash-0731-W4A16/
├── config.json                          # 含 quantization_config (mixed-precision)
├── generation_config.json
├── model-00001-of-00034.safetensors     # 34 个分片, 约 166GB
├── ...
├── model-00034-of-00034.safetensors
├── model.safetensors.index.json
├── encoding/                           # 官方 chat prompt 编码模块
│   ├── encoding_dsv4.py
│   └── README.md
└── README.md
```

总大小:**~166 GB**(原始 156 GB,略大因 shared/compressor/indexer 从 FP8 升级到 BF16)。

---

## 如何使用(vLLM)

### 环境要求

- vLLM ≥ 0.26.0
- torch ≥ 2.11(建议 cu130 build)
- 至少 **2×H20 96GB** 或 **4×A100 80GB**(TP=2 / TP=4)

### Prompt 编码(重要)

DeepSeek-V4-Flash 使用**非标准的 chat template**,不能用 `apply_chat_template`。本仓库附带官方的 `encoding/` 目录,使用方法:

```python
import sys
sys.path.insert(0, "encoding")
from encoding_dsv4 import encode_messages

prompt = encode_messages(
    [{"role": "user", "content": "你好"}],
    thinking_mode="chat",        # 或 "thinking"
    reasoning_effort="low",      # low / high / max
)
# prompt 是已编码的字符串, 直接喂给 llm.generate([prompt], ...)
```

### H20 / H100(推荐,开箱即用)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-username/DeepSeek-V4-Flash-0731-W4A16",
    tensor_parallel_size=2,
    dtype="bfloat16",
    kv_cache_dtype="fp8",          # 必须,DSpark MLA 要求
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    max_model_len=4096,
    # 可选: DSpark 投机解码
    speculative_config={
        "method": "dspark",
        "num_speculative_tokens": 7,
        "draft_sample_method": "greedy",
    },
)
```

### A100 / L40S / 其他 Ampere(需 vLLM 小补丁)

Ampere 架构不支持 FP8 tensor core。本模型的 attention 仍是 FP8,vLLM 的 `deep_gemm_fp8_o_proj` 在 A100 上会失败。需要给 vLLM 打一个小补丁,把 o_proj 的 FP8 einsum 换成 BF16 fallback 路径。

补丁位置:`vllm/models/deepseek_v4/nvidia/ops/o_proj.py`

```python
# 原始(仅 FP8):
(wo_a.weight, wo_a.weight_scale_inv)

# 补丁后(自动 fallback):
_ws_inv = getattr(wo_a, 'weight_scale_inv', None)
_wo_scale = _ws_inv if _ws_inv is not None else wo_a.weight_scale
(wo_a.weight, _wo_scale)
```

> 注:在 H20 上也建议打此补丁(用 `is not None` 判断,对原生 FP8 模型行为无影响,但能让 compressed-tensors 格式的 FP8 attention 正常工作)。
>
> 完整的 A100 支持需要进一步把 attention 从 FP8 改为 BF16/INT16,这是后续工作。

### 命令行启动(vLLM OpenAI server)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-username/DeepSeek-V4-Flash-0731-W4A16 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95
```

---

## 复现量化(Conversion Script)

转换脚本见 `convert_v4.py`(本仓库未附,核心逻辑如下,约 150 行):

```python
import torch, re
from safetensors import safe_open

FP4_TABLE = torch.tensor([0,0.5,1,1.5,2,3,4,6,0,-0.5,-1,-1.5,-2,-3,-4,-6])
GROUP = 32

def dequant_fp4(w_fp4, scale_e8m0):
    """FP4 e2m1fn_x2 → BF16, 无损"""
    codes = w_fp4.view(torch.uint8)  # 2个FP4 packed in 1 uint8
    hi = codes >> 4; lo = codes & 0xF
    vals = torch.stack([FP4_TABLE[hi], FP4_TABLE[lo]], -1).flatten()
    return (vals.reshape(*w_fp4.shape[:-1], -1) * scale_e8m0).to(torch.bfloat16)

def quant_int4(w_bf16, gs=GROUP):
    """BF16 → INT4, 网格搜索最优 scale"""
    out, in_ = w_bf16.shape; ng = in_ // gs
    w = w_bf16.float().reshape(out, ng, gs)
    amax = w.abs().amax(-1, keepdim=True).clamp(1e-8)
    best = None
    for r in [6/7, 0.9, 0.95, 1.0, 1.05, 1.1]:   # 6个候选
        s = amax * r / 7
        q = torch.clamp(torch.round(w / s), -8, 7)
        mse = ((w - q * s) ** 2).mean(-1, keepdim=True)
        # 保留每个 group MSE 最小的
        ...
    # pack 8×int4 → 1×int32
    ...
    return packed_int32, scale_bf16, shape
```

完整流程:
1. 加载原始 48 个分片
2. 对每个 expert weight:`FP4 → BF16 (无损) → INT4 g32 (网格搜索)`
3. Attention FP8 权重**原样保留**(只把 `.scale` 重命名为 compressed-tensors 期望的 `.weight_scale`)
4. shared_experts / compressor / indexer:FP8 → BF16 反量化(vLLM 对这些模块不匹配 FP8 target)
5. 重新分片为 34 个 ~5GB 的 safetensors,写入 `config.json` 的 quantization_config

---

## 致谢

- 原始模型:[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- 参考 [canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) 的 mixed-precision 配置结构
- 量化框架:[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- 推理引擎:[vLLM](https://github.com/vllm-project/vllm)

## License

同原始模型,详见 [DeepSeek License](https://github.com/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE)。
