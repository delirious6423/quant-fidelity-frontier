---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen4_exp
- moe
- sparse-attention
- ngram-embedding
- uncensored
---

# SamQuant C58 · Qwen3.8-Flash-Next 量化包（为 128GB Mac 制作）

一个在 **Apple M3 Max / 128GB 统一内存**上手工量化出的 Qwen3.8-Flash-Next（qwen4_exp 架构，180B 级 MoE：125B 主干 + 51B n-gram 表 + 4B MTP 投机头）的 mlx-serve 混合位宽包：常驻约 60GB，无审查源，带视觉塔。

```bash
# 方式一：引擎直接拉取（首次会下载整库 ~92GB）
mlx-serve --model LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 --serve

# 方式二：先下到本地再挂
hf download LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 --local-dir ~/models/samquant-c58
mlx-serve --model ~/models/samquant-c58 --serve
```

> 注意：Hugging Face 不允许库名含双连字符，所以本仓库名是单 `-` 版；包内容与其它 SamQuant 包完全同名对应。
> 依赖：本包含 3-bit/6-bit 权重，**需要 mlx-serve 26.9.1 及以上**才能正确解包（旧版会解成噪声）。
> `ngram_table.bin` 是独立 mmap 的 4-bit n-gram 表（32,000,153,976 字节），必须与分片同目录、保持 .bin 后缀，不要重命名或转换它。

## 📖 公众号原文（送包说明 · 实测全过程 · 中文）

电脑端直接点链接：**https://mp.weixin.qq.com/s/G1h72LdFwRLdaP6W7yIIYg**（《亲测：520k上下文跑出46 tok/s、无审查、常驻61GB、连跑51小时零失败——128G Mac自量化包》）
手机端微信扫码看原文：

![公众号文章二维码](images/wechat-article-qr.png)

## 这个包是怎么来的（功劳摆正）

- 基座模型：Qwen 官方 Qwen3.8-Flash-Next；原始权重：社区团队 orcarouter 的无审查 BF16 包（335.3 GiB / 131 分片，逐文件 sha256 验收 132/132 通过）
- 量化工具链：mlx-serve 仓库自带产线（imatrix 校准采集 → 贪心位宽分配 → 混合宽度转换 → 留出窗口打分），感谢上游作者
- 本包作者做的四件事：**用自己的 AI 会话日志做校准语料**（三个 agent 工具的真实流量 + 官方通用语料，按内容 sha1 去重 26.11%，密钥/凭证脱敏）、逐层贪定位宽配方、跑完整产线、做质量与业务验收
- 卡片样式沿用 mlx-serve 生态包卡片惯例（模板参考 ddalcu 上游卡片），读数全部替换为 M3 Max 128GB 本机实测

## 位宽配方（C58 档）

| 部件 | 位宽/组宽 |
|---|---|
| 路由专家（512×48 层，约 121B） | 0-45 层逐层贪心：3-bit g64 × 70 组 + 4-bit g64 × 26 组（imatrix 加权误差测量，按字节预算贪心分配）；46-47 层与 MTP 头专家 4-bit g64 |
| 注意力 / GDN / Hyper-Connections / indexer / 共享专家 | 6-bit g64 |
| lm_head | 8-bit g64 |
| embed_tokens | 4-bit g64 |
| n-gram 表 | 4-bit g32（行宽 160，独立 ngram_table.bin，mmap 常驻不进 GPU） |
| routers / inject gates / norms / convs / SSM state | bf16 |
| 视觉塔 | 密集 bf16（model-vision.safetensors，~0.9GB） |

整库约 92GB：权重与视觉塔 59.83GB（索引载荷口径，100 个编号分片 + model-vision.safetensors）+ 独立 n-gram 表 32,000,153,976 字节（ngram_table.bin）+ 9 个配置/文档文件。

## 本机实测（M3 Max 128GB，mlx-serve 26.9.6 / MLX 0.32.2）

**生产运行口径**（日常 DSH agent 会话，2026-10-06 17:24 起 51 小时）：
- 完成 947 轮，失败 0（6 次为人工中断）；缓存命中 98.19%
- decode 中位 36.9 tok/s（p10 26.2 / p90 52.6），单轮最大上下文 421,473 token
- 加载后进程实测占用（phys_footprint）61GB

**实跑快照（2026-10-10 12:31，本包日常承担公众号文章会话时）**：

![引擎实时仪表盘](images/live-metrics-20261010.png)

仪表盘瞬时值：decode 34.1 tok/s（落在上表 p10~p90 区间内）、缓存命中 98%（1214/1242 查询）、GPU 96%。注意 MEMORY 格 86,758 MB 是"此刻"物理占用——当条会话已吃进约 283K token 上下文，常驻随对话增长（加载完即时值约 61GB）；128GB 机器余量充足。

![DSH 会话现场](images/session-cockpit-20261010.png)

同刻的宿主会话状态：一条 16 分 59 秒的深研任务在跑，上下文 283K/524K（54%），本会话 decode 31 tok/s、累计 70.7M token 缓存命中 98%。

（images/ 目录为说明示例图，非模型文件；核对模型完整性以 model.safetensors.index.json 引用的 101 个文件 + ngram_table.bin 口径为准。）

**专项窗口口径**（2026-09-30，524K ctx，57 轮 >30K prompt 的 agent 实战）：
- decode p50 46.0 tok/s；同晚同协议的更轻同系列包 B55（52.4GB 权重）p50 为 37.0——本包更重反而更快，因 4-bit 专家落在引擎融合内核快速通道内

**离线质量口径**（600 个留出打分位，top-1 与 BF16 原版一致率，打分前缀 65~2,297 token）：
- C58 85.40%；对照社区 4-bit 包 87.82%（−2.42 分）；同预算全 4-bit 包 Ap73 85.76%（+0.36 分）

**业务验收**（24 题 ×7 包：工具编排/代码/长文档各 8，2026-10-04）：
- 本包：长文 0 失败、代码吞吐 50.4 tok/s 为七包最高；工具编排题 3/8——多轮工具编排是这批量化包的共同短板，请按需预期

## 推荐引擎参数（与生产运行一致）

```bash
mlx-serve --model <本包路径> --serve --host 127.0.0.1 --port 8093 --metrics \
  --ctx-size 524288 \
  --config-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144},"max_position_embeddings":524288}}' \
  --prefix-cache-mem 32GB --prefix-cache-entries 128 \
  --mtp --kv-quant 4 --pld-draft-len 8 --mtp-head-kv-quant \
  --prefill-chunk 8192 --reasoning-budget 32768 \
  --idle-evict-secs 3600 --tokenize-cache-entries 16 \
  --os-reserve-gib 24 --wired-margin-gib 8 --timeout 0
```
（64GB 机器请改用同系列 B55 档；335GB 原始 BF16 与产线过程另见公众号「悟也空空」。）

## 诚实的边界

- 以上所有读数仅代表这台 128GB M3 Max；离线打分的前缀偏短（65~2,297 token），长上下文质量未做专门验证；top-1 一致率与真实业务能力存在脱钩案例（详见作者下一篇），选型请以自己的任务实测为准。
- 模型权重许可证 qwen-community-1.0（LICENSE 随包）；"无审查"属性继承自 orcarouter 原料，本包未做任何行为修改。
- 本仓库为爱好者自制量化包，非官方出品。

## 联系

公众号「悟也空空」留言/私信，或评论区交流你的 128G Mac 使用数据。
