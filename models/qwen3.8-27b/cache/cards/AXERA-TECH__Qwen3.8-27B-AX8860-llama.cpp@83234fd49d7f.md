---
base_model:
- Qwen/Qwen3.8-27B
---
# Qwen3.8-27B on AX8860 (llama.cpp + ggml-axera)

在爱芯 AX8860 的 NPU 上用 llama.cpp 运行 Qwen3.8-27B：文本 + 图像输入，最长 256k 上下文，MTP 投机解码，OpenAI 兼容服务。
Qwen3.8-27B on the Axera AX8860 NPU with llama.cpp: text + image input, up to 256k context, MTP speculative decoding, OpenAI-compatible server.

## 性能 / Performance

| | |
|---|---|
| decode | **13.8 tokens/s** |
| decode，MTP 投机解码 / with MTP | 最高 / up to **33.0 tokens/s** |
| 4k prompt prefill | **1249 tokens/s**（3908 token 3.13 s） |
| 上下文 / context | 最长 / up to 262144 |

MTP（`MTP=1`）按内容 / by content：

| 内容 / content | 普通 / plain | MTP | 加速 / speed-up |
|---|---:|---:|---:|
| 代码 / code | 13.9 | **33.0** | 2.38x |
| 中译英 / translation | 13.8 | **26.8** | 1.94x |
| Markdown 表格 / table | 13.8 | **30.8** | 2.24x |
| 数学解题 / maths | 13.8 | **30.9** | 2.23x |
| 中文短文 / Chinese essay | 13.8 | **19.2** | 1.39x |
| 英文故事 / English story | 13.8 | **21.2** | 1.54x |
| 代码，推荐采样 / code, sampled | 13.7 | **28.4** | 2.07x |
| 中文短文，推荐采样 / essay, sampled | 13.7 | **18.9** | 1.38x |

单位 tokens/s；每项最多生成 256 token，未注明的为贪心解码 / tokens/s; up to 256 tokens each, greedy unless noted.

`sh bench.sh` 复现 / reproduce（生成 128 token / 128 generated tokens）：

| prompt | `-c` | prompt 耗时 / time | decode tokens/s |
|---|---:|---:|---:|
| 5 token | 8192 | 0.24 s | 13.8 |
| 47 token | 8192 | 0.25 s | 13.8 |
| 3908 token | 8192 | 3.13 s | 13.5 |
| 3908 token | 16384 | 3.13 s | 13.5 |
| 3908 token，推荐采样 / sampled | 8192 | 3.12 s | 13.3 |

## 快速开始 / Quick start

```sh
sh run.sh "The capital of France is" 64           # 文本 / text
sh run.sh -i bus.jpg "Describe this image." 256   # 图像 / image
sh run.sh -c                                      # 命令行对话 / chat
sh run.sh -s 8080                                 # OpenAI 兼容服务 / server
MTP=1 sh run.sh -s 8080                           # MTP 投机解码 / speculative decoding
CTX=131072 sh run.sh -s 8080                      # 长上下文 / long context (default 16384)
sh bench.sh                                       # 性能复现 / benchmark
```

需要 AX8860，SDK V0.5.1 或更新；HF 下载的文件没有执行权限，用 `sh` 调用即可 / needs an AX8860 with SDK V0.5.1+; call the scripts with `sh`.

## 文件 / Files

| 文件 / File | 说明 / Description |
|---|---|
| `qwen3.8-27b-mxfp4.gguf` | 模型权重 / weights，14.8 GB，md5 `32f5a603d4f75edb16ccf7ff3a35bdd6` |
| `mmproj-qwen3.8-27b-bf16.gguf` | 视觉编码器 / vision encoder，931 MB，md5 `d2b989d06a0eb9109bab182ec2ff03f2` |
| `bin/` | llama.cpp 板端程序（静态链接）/ board programs (static) |
| `kernels/` | NPU kernel + `manifest.json` |
| `run.sh`，`chat_template.jinja` | 运行脚本与服务端对话模板 / run script and the server's chat template |
| `bench.sh` 及 prompt | 性能复现 / benchmark |

## 说明 / Notes

- 服务端默认使用模型卡推荐的采样参数，文本补全与图像描述为贪心解码 / the server uses the model card's recommended sampling; completion and image description are greedy.
- 服务端默认关闭思考（`--reasoning off`），并允许对话中段的 system 消息（Codex 等 agent 客户端可用）/ the server does not think by default and accepts a mid-conversation system message (works with agent clients such as Codex).
- Prompt cache 默认放在 4 GB 内存；设置 `CACHE_DIR=<SSD 目录>` 可落盘，重启后仍能命中 / the prompt cache lives in 4 GB of host memory; set `CACHE_DIR=<an SSD directory>` to keep it on disk across restarts.
- MTP 每轮猜 3 个 token，由目标模型逐个验证；贪心时输出与普通解码一致 / MTP drafts 3 tokens per round, each verified by the target model; greedy output equals plain decoding.
- 运行时独占 NPU 4 核；服务端 `-np 1` / takes all 4 NPU cores; the server runs `-np 1`.
- `llama-bench` 可用，请加 `-b 4096 -ub 4096` / `llama-bench` works; pass `-b 4096 -ub 4096`.
- 个别环境下连续请求偶发 `ret = -3`，重试即可 / in rare cases a back-to-back request fails with `ret = -3`; retry it.

## 正确性 / Correctness

- "The capital of France is" 的贪心输出与上一版逐字节相同；图像描述正确（车辆、文字、人物）/ the greedy output of "The capital of France is" is byte-identical to the previous release; image descriptions correct.
- 中文 PPL（内部测试集，`-c 512 --chunks 2`）：prefill 8.84，逐 token decode（`-b 1`）8.83，投机解码的 4 行验证（`-ub 4`）8.82；上一版同条件 prefill 8.83，decode 8.73 / Chinese PPL on an internal set: 8.84 through the prefill, 8.83 token by token, 8.82 through the 4-row verification; the previous release: 8.83 / 8.73.
- 投机解码的输出与普通贪心解码一致（每个草稿 token 都经目标模型验证）/ speculative decoding gives the same output as plain greedy decoding (every draft token is verified by the target model).

## 源码 / Sources

- llama.cpp：[ZHEQIUSHUI/llama.cpp](https://github.com/ZHEQIUSHUI/llama.cpp) `axera-dev` @ `aea9714`
- kernel：[ZHEQIUSHUI/ax-dsl-kernels](https://github.com/ZHEQIUSHUI/ax-dsl-kernels) `main` @ `cd3394b`（`models/qwen38_27b`）

## 更新 / Changelog

- 2026-11-07：prefill 提速（4k 约 1250 tokens/s）；MTP 改为可选（`MTP=1`），`CTX` 默认 16384；服务端兼容中段 system 消息、默认关闭思考、prompt cache 可落盘到 SSD；NPU 采样；新增 `bench.sh` / server accepts mid-conversation system messages, no thinking by default, prompt cache on SSD; NPU sampling; `bench.sh`.