---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: gguf
language:
- en
- zh
tags:
- gguf
- qwen3.8
- qwen3.5
- uncensored
- llama.cpp
- sakura
---

![Qwen3.8-27B Uncensored](uncensored-banner.png)

# Qwen3.8-27B Uncensored (GGUF, 16 GB class)

Qwen3.8-27B with its refusal behaviour reduced, as a compact file for 16 GB GPUs. Apart from that it is the original model. This is an independent community release, not by Qwen or Alibaba.

- File: `Qwen3.8-27B-Uncensored-IQ3_S-16GB.gguf` (13.5 GB, includes the MTP tensors)
- Context: tested at 32,768 tokens
- Chat template: the one embedded in the file; thinking on by default

## Benchmarks of the original model

These are the numbers the Qwen team publishes for the unmodified Qwen3.8-27B (see the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the full table, comparison models and evaluation settings). They were **not re-measured here**; the change and the much smaller quantization can shift individual results.

| Benchmark | Qwen3.8-27B (original, as reported by Qwen) |
|---|---|
| Terminal Bench 2.1 (Terminus) | 73.0 |
| SWE-bench Pro | 61.7 |
| IFBench | 79.5 |
| GPQA Diamond | 89.2 |
| HLE | 30.8 |
| LiveCodeBench v6 | 90.3 |

## Our measurements

**Refusals.** 104 held-out test prompts of the public `mlabonne/harmful_behaviors` set, greedy decoding, thinking off, at most 100 new tokens. "Direct refusals" count answers that contain a plain refusal phrase ("I cannot", "I can't", "I won't", "I will not", "sorry", "I'm unable"). "Any refusal or disclaimer phrase" additionally counts words such as "illegal", "violat…", "prohibit", "harmful" and "as an AI", which also appear in answers that carry a disclaimer but no refusal. The comparison model is plain Qwen3.8-27B (Q4_K_M, ggml-org).

| Model | Direct refusals (of 104) | Any refusal or disclaimer phrase (of 104) | Degenerate answers |
|---|---|---|---|
| Qwen3.8-27B (Q4_K_M, ggml-org) | 103 | 103 | not measured |
| **This model** | **0** | **89** | 0 |

This is a phrase-based count, not a safety evaluation and not a measure of how useful the answers are: the model stops refusing directly, but many of its answers still carry warnings. The answer texts were not stored.

**Coding check.** The same hard algorithmic coding tasks (frozen local set, hidden tests) and settings for both rows: seed 42 + task number, temperature 0.6, top-p 0.95, top-k 20, at most 16,384 output tokens, thinking on, context 32,768; llama.cpp b12189 (CUDA) on one NVIDIA L40S. "Solved" means the answer was finished and the code passed all tests; "limit N" means the 16,384-token limit was hit without a passing solution; "pass at limit" means it hit the limit but code found in the output passed (not counted as solved). One seed and five tasks only (the sixth task of the usual set was left out because no model solves it): a small sample, and the two files use different quantizations, so this shows that the model still works, not that it is better or worse.

| Model | Solved (finished) | Passed incl. at limit | Output tokens (sum) | Task 27 | Task 10 | Task 8 | Task 9 | Task 11 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (Q4_K_M, ggml-org) | 1 / 5 | 4 / 5 | 66,767 | pass at limit 16384 | pass at limit 16384 | pass at limit 16384 | limit 16384 | pass 1231 |
| **This model** | 1 / 5 | 3 / 5 | 67,011 | pass at limit 16384 | limit 16384 | pass at limit 16384 | limit 16384 | pass 1475 |

## Quick start

```bash
llama-server -hf webmp3/Sakura-Qwen3.8-27B-Uncensored-GGUF --jinja -c 16384 -ngl 99 --temp 0.6 --top-p 0.95 --top-k 20
```

Then open http://127.0.0.1:8080 or use the OpenAI-compatible endpoint `http://127.0.0.1:8080/v1`. On a 16 GB card keep the context at 16384 or lower (see below). Use a current llama.cpp build with `qwen35` support. Ollama (`ollama run hf.co/webmp3/Sakura-Qwen3.8-27B-Uncensored-GGUF`) and LM Studio (Discover tab, search for `webmp3/Sakura-Qwen3.8-27B-Uncensored-GGUF`) have not been tested by the author.

## Size and quantization

| File | Quantization | Size | Average bits per weight |
|---|---|---|---|
| `Qwen3.8-27B-Uncensored-IQ3_S-16GB.gguf` | Mixed i-quant/K-quant: IQ3_XXS 49 %, Q8_0 29 %, IQ4_XS 9 %, Q6_K 8 %, IQ2_XXS 6 % of the file | 13.5 GB (12.55 GiB) | about 3.9 |

Only this one file is published.

## Hardware compatibility

- **Memory:** the weights take 13.5 GB. Add roughly 1 GB of cache per 16k tokens of context plus about 1 GB of compute buffers (an estimate, not a measured figure). A 16 GB card therefore fits about 8k to 16k of context; at 32,768 tokens plan for 17 to 18 GB. With less memory, offload part of the layers and keep the rest in system RAM (slower).
- **Tested:** NVIDIA L40S (48 GB) with llama.cpp b12189 (CUDA), fully offloaded, 32,768 context.
- **Not tested:** 16 GB cards themselves, CPU-only inference, Apple Silicon, other GPUs.

## Responsible use

Safety behaviour is reduced by design. The model is meant for adults doing research, creative work and local experimentation. You are responsible for how you use it and for complying with the law where you live.

## License and attribution

Apache-2.0, derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). The quantization this file is built on is by ByteShape ([byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF), Apache-2.0).

<details>
<summary>中文说明 · 樱花 Sakura</summary>

# Qwen3.8-27B 无审查版（GGUF，16 GB 级）

降低了拒答行为的 Qwen3.8-27B，提供适合 16 GB 显卡的紧凑文件。除此之外就是原始模型。这是独立的社区发布，并非 Qwen 或阿里巴巴官方。

- 文件：`Qwen3.8-27B-Uncensored-IQ3_S-16GB.gguf`（13.5 GB，包含 MTP 张量）
- 上下文：在 32,768 token 下测试
- 聊天模板：文件内置；默认开启思考

## 原始模型的基准成绩

以下数字是 Qwen 团队为未经修改的 Qwen3.8-27B 公布的成绩（完整表格、对比模型和评测设置见[原始模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)）。它们**未在此重新测量**；修改和小得多的量化可能使个别结果有所变化。

| 基准 | Qwen3.8-27B（原始模型，Qwen 官方公布） |
|---|---|
| Terminal Bench 2.1 (Terminus) | 73.0 |
| SWE-bench Pro | 61.7 |
| IFBench | 79.5 |
| GPQA Diamond | 89.2 |
| HLE | 30.8 |
| LiveCodeBench v6 | 90.3 |

## 我们的测量

**拒答。** 使用公开数据集 `mlabonne/harmful_behaviors` 的 104 条留出测试提示，贪心解码，关闭思考，最多生成 100 个 token。“直接拒答”统计包含明确拒绝用语（“I cannot”“I can't”“I won't”“I will not”“sorry”“I'm unable”）的回答。“任何拒答或免责声明用语”还包括 “illegal”“violat…”“prohibit”“harmful”“as an AI” 等词，这些词也会出现在带有免责声明但并未拒绝的回答中。对比模型是原版 Qwen3.8-27B（Q4_K_M，ggml-org）。

| 模型 | 直接拒答（共 104） | 任何拒答或免责声明用语（共 104） | 退化回答 |
|---|---|---|---|
| Qwen3.8-27B（Q4_K_M，ggml-org） | 103 | 103 | 未测量 |
| **本模型** | **0** | **89** | 0 |

这是基于用语的计数，并非安全评估，也不衡量回答是否有用：模型不再直接拒绝，但许多回答仍带有警告。未保存回答文本。

**编程检查。** 两行使用相同的高难度算法编程题（固定的本地题集，隐藏测试）和相同设置：种子 42 + 题号，temperature 0.6，top-p 0.95，top-k 20，最多输出 16,384 token，开启思考，上下文 32,768；在一块 NVIDIA L40S 上使用 llama.cpp b12189（CUDA）。“已解决”指回答完整结束且代码通过全部测试；“达上限 N”表示触及 16,384 token 上限且没有通过的解答；“超限通过”表示触及上限，但在输出中找到的代码通过了测试（不计为已解决）。仅一个种子、五道题（常规题集的第六题因无模型能解而省略）：样本很小，且两个文件的量化不同，因此这只说明模型仍可正常工作，并不说明更好或更差。

| 模型 | 已解决 | 含超限通过 | 输出 token 总数 | 任务 27 | 任务 10 | 任务 8 | 任务 9 | 任务 11 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (Q4_K_M, ggml-org) | 1 / 5 | 4 / 5 | 66,767 | 超限通过 16384 | 超限通过 16384 | 超限通过 16384 | 达上限 16384 | 通过 1231 |
| **本模型** | 1 / 5 | 3 / 5 | 67,011 | 超限通过 16384 | 达上限 16384 | 超限通过 16384 | 达上限 16384 | 通过 1475 |

## 快速开始

```bash
llama-server -hf webmp3/Sakura-Qwen3.8-27B-Uncensored-GGUF --jinja -c 16384 -ngl 99 --temp 0.6 --top-p 0.95 --top-k 20
```

然后打开 http://127.0.0.1:8080，或使用兼容 OpenAI 的接口 `http://127.0.0.1:8080/v1`。在 16 GB 显卡上请将上下文保持在 16384 或更低（见下文）。请使用支持 `qwen35` 的较新 llama.cpp 版本。Ollama（`ollama run hf.co/webmp3/Sakura-Qwen3.8-27B-Uncensored-GGUF`）和 LM Studio（Discover 标签页，搜索 `webmp3/Sakura-Qwen3.8-27B-Uncensored-GGUF`）作者未测试。

## 大小与量化

| 文件 | 量化 | 大小 | 平均每权重位数 |
|---|---|---|---|
| `Qwen3.8-27B-Uncensored-IQ3_S-16GB.gguf` | i-quant/K-quant 混合：文件的 IQ3_XXS 49 %、Q8_0 29 %、IQ4_XS 9 %、Q6_K 8 %、IQ2_XXS 6 % | 13.5 GB（12.55 GiB） | 约 3.9 |

仅发布这一个文件。

## 硬件兼容性

- **显存/内存：** 权重占 13.5 GB。每 16k token 上下文大约再加 1 GB 缓存，另加约 1 GB 计算缓冲区（估计值，并非实测）。因此 16 GB 显卡可容纳约 8k 至 16k 上下文；使用 32,768 token 时请预留 17 至 18 GB。显存不足时只卸载部分层，其余放在系统内存中（速度更慢）。
- **已测试：** NVIDIA L40S（48 GB），llama.cpp b12189（CUDA），完全卸载，上下文 32,768。
- **未测试：** 16 GB 显卡本身、纯 CPU 推理、Apple Silicon、其他 GPU。

## 负责任使用

安全行为被有意降低。本模型面向成年人的研究、创作和本地实验。你需对使用方式负责，并遵守所在地的法律。

## 许可证与署名

Apache-2.0，源自 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)（Apache-2.0）。本文件所基于的量化由 ByteShape 提供（[byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF)，Apache-2.0）。

完整独立的中文版：[README_zh.md](README_zh.md)

</details>
