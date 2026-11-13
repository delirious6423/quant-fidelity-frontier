---
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
---

# Qwen3.8-27B on AX8860 (llama.cpp + ggml-axera)

在爱芯 AX8860 的 NPU 上用 llama.cpp 运行 Qwen3.8-27B：文本 + 图像输入，最长 256k 上下文，可选 MTP 投机解码，OpenAI 兼容服务。
Qwen3.8-27B on the Axera AX8860 NPU with llama.cpp: text + image input, up to 256k context, optional MTP speculative decoding, OpenAI-compatible server.

## 性能 / Performance

| | |
|---|---|
| decode | **13.8 tokens/s** |
| decode，MTP 投机解码 / with MTP | 最高 / up to **35.1 tokens/s**（2.52x） |
| 4k prompt，MTP / MTP on 4k | 13.5 → **20.8 tokens/s**（1.54x，accepted 32/90） |
| 4k prompt prefill | **1262 tokens/s**（3908 token 3.10 s） |
| 上下文 / context | 最长 / up to 262144 |

MTP（`MTP=1`）按内容 / by content:

| 内容 / content | 普通 / plain | MTP | 加速 / speed-up | 接受率 / accept |
|---|---:|---:|---:|---:|
| 代码 / code | 13.9 | **35.1** | 2.52x | 0.785 |
| 中译英 / translation | 13.7 | **28.6** | 2.09x | 0.596 |
| Markdown 表格 / table | 13.8 | **31.0** | 2.25x | 0.687 |
| 数学解题 / maths | 13.8 | **32.0** | 2.32x | 0.709 |
| 中文短文 / Chinese essay | 13.7 | **20.6** | 1.51x | 0.342 |
| 英文故事 / English story | 13.7 | **20.4** | 1.49x | 0.338 |
| 代码，推荐采样 / code, sampled | 13.7 | **29.0** | 2.12x | 0.618 |
| 中文短文，推荐采样 / essay, sampled | 13.7 | **20.0** | 1.46x | 0.322 |

单位 tokens/s；每项最多生成 256 token，未注明的为贪心解码 / tokens/s; up to 256 tokens each, greedy unless noted.

`sh bench.sh` 复现 / reproduce（生成 128 token / 128 generated tokens；短 prompt 仅列耗时 / short prompts report time only）：

| prompt | `-c` | prompt 耗时 / time | decode tokens/s |
|---|---:|---:|---:|
| 5 token | 8192 | 0.22 s | — |
| 47 token | 8192 | 0.22 s | — |
| 3908 token | 8192 | 3.10 s | 13.6 |
| 5 token | 16384 | 0.21 s | — |
| 47 token | 16384 | 0.22 s | — |
| 3908 token | 16384 | 3.09 s | 13.6 |
| 3908 token，推荐采样 / sampled | 8192 | 3.09 s | 13.3 |

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

- MTP 为可选：`MTP=1 ./run.sh -s` 显式开启，服务端不再用自适应门控把它关掉；不设置 `MTP` 时保持普通解码。贪心请求请显式带 `presence_penalty: 0` / MTP is opt-in: `MTP=1 ./run.sh -s` enables it explicitly and bypasses the adaptive gate; without `MTP` the server stays non-speculative. Greedy requests should set `presence_penalty: 0`.
- 服务端默认使用模型卡推荐的采样参数，但 presence penalty 默认为 0（模型卡的 1.5 会让模型回避复述提示里出现过的内容，如文件名、数字、代码，影响 agent 与工具调用）；`PRESENCE_PENALTY=1.5 ./run.sh -s` 可恢复，请求里自带的 `presence_penalty` 优先。文本补全与图像描述为贪心解码 / the server uses the model card's recommended sampling, except that the presence penalty defaults to 0 (the card's 1.5 makes the model avoid repeating text from the prompt such as file names, numbers and code, which hurts agents and tool calls); `PRESENCE_PENALTY=1.5 ./run.sh -s` restores it, and a request's own `presence_penalty` wins. Completion and image description are greedy.
- 服务端默认关闭思考（`--reasoning off`），并允许对话中段的 system 消息（Codex 等 agent 客户端可用）/ the server does not think by default and accepts a mid-conversation system message (works with agent clients such as Codex).
- Prompt cache 默认放在 4 GB 内存；设置 `CACHE_DIR=<SSD 目录>` 可落盘，重启后仍能命中 / the prompt cache lives in 4 GB of host memory; set `CACHE_DIR=<an SSD directory>` to keep it on disk across restarts.
- 运行时独占 NPU 4 核；服务端 `-np 1` / takes all 4 NPU cores; the server runs `-np 1`.
- `llama-bench` 可用，请加 `-b 4096 -ub 4096` / `llama-bench` works; pass `-b 4096 -ub 4096`.
- 个别环境下连续请求偶发 `ret = -3`，重试即可 / in rare cases a back-to-back request fails with `ret = -3`; retry it.
- MTP 每轮猜 3 个 token，由目标模型逐个验证；贪心时输出与普通解码一致 / MTP drafts 3 tokens per round, each verified by the target model; greedy output equals plain decoding.
- 长上下文使用滚动 checkpoint（默认 `-ctxcp 4`），内存不随上下文线性增长 / long context uses rolling checkpoints (default `-ctxcp 4`); memory does not grow linearly with context.

## 正确性 / Correctness

- "The capital of France is" 的贪心输出与上一版逐字节相同；图像描述正确 / the greedy output of "The capital of France is" is byte-identical to the previous release; image descriptions correct.
- MTP 默认关闭（`MTP=1` 开启）；开启后输出由目标模型逐 token 验证，与普通贪心解码一致 / MTP is off by default (`MTP=1` enables it); every draft is verified by the target model and greedy output equals plain decoding.
- `r3client.py grow` 到 ≥60k prompt token 不 OOM，checkpoint 内存平稳 / `r3client.py grow` reaches ≥60k prompt tokens without OOM and checkpoint memory stays flat.

## 源码 / Sources

- llama.cpp：[ZHEQIUSHUI/llama.cpp](https://github.com/ZHEQIUSHUI/llama.cpp) `axera-dev` @ `0104880be`

## 更新 / Changelog

- 2026-11-11：服务端未指定模板时自动使用包内 `chat_template.jinja`（手动启动的服务不再让 agent 客户端收到 HTTP 500）；服务模式 presence penalty 默认改为 0 / the server uses the package's `chat_template.jinja` when no template is given (a hand-started server no longer answers agent clients with HTTP 500); the server's default presence penalty is now 0.
- 2026-11-09：统一板端 bin；MTP 验证逻辑修复；贪心场景显式 `presence_penalty: 0`；perplexity 固定走全 logits / unified board bins; MTP verification fixes; explicit `presence_penalty: 0` in greedy scenes; perplexity keeps the full-logits head.

