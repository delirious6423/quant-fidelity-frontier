---
language: [en, zh]
base_model:
  - Qwen/Qwen3.8-27B
  - prism-ml/Ternary-Bonsai-2-27B-gguf
  - ukisai/Swift-Bonsai-2-GGUF
library_name: llama.cpp
pipeline_tag: text-generation
license: apache-2.0
tags: [Qwen3.8-27B, Qwen, 27B, quantization, GGUF, MTP, speculative-decoding, llama.cpp, local-inference, code-generation]
---

# Qwen3.8-27B-Flash-MTP — a 27B ternary code model with MTP self-speculation, on an 8 GB card

**A 27B code model that fits entirely in 8 GB of VRAM and stays fast while you type.**

No rented API, no cloud, no datacentre GPU. A 27B-parameter model, ternary-quantised,
with a self-speculative draft head packed into the same file — running on a consumer
laptop GPU, at interactive speed.

| Workload | Typical speed |
|---|---|
| **Code generation** — Python, C, SQL, Go, TS, Rust, JS | **~70–90 t/s** |
| **Editing mode** — revise an existing file or document | **~100–200 t/s** |

Measured on an **RTX 4060 Laptop, 8 GB**, with thinking off, driving `llama-server` from
this project's fork in real development work. Treat these as the usual range rather
than a guarantee: how fast you land depends on how much of your output is text an
ngram drafter can predict, which is why code and editing beat open-ended prose here.
Your own numbers will land somewhere in this neighbourhood.

**88K of context, fully in VRAM.** With `q3_0` — a 3-bit K/V cache, 14 KB per token
against q4_0's 18 KB — the same 8 GB card keeps an 88K-token context entirely in VRAM.
As far as we know, this is the first time a 27B-class model runs a full long-context
agent session at this speed on consumer hardware.

For comparison, the same base model at official 4-bit quantisation runs at roughly
**1–2 t/s** on the same class of machine.

---

## Why it is fast

27B goes from ~54 GB (FP16) to **5.79 GB** by stacking four things:

| Technique | Effect | Where it comes from |
|---|---|---|
| **Ternary quantisation (PTQ1_0) + CUDA kernels** | 27B in 5.79 GB, fully resident | PrismML Bonsai-2 backbone + this fork's kernels |
| **CUDA graphs** | ~1.5× | llama.cpp / ggml |
| **MTP speculative decoding, depth ≤4** | several tokens per forward pass | this fork — the draft head is **inside the gguf** (`blk.64.nextn`) |
| **24k draft-only vocabulary shortlist** | 248k → 24k rows read per draft step, **~90% fewer bytes** | **our own optimisation** |

That last one is the interesting part. The MTP draft head's `output.weight` is a
full-vocabulary `[5120, 248320]` matrix — 265 MB — and the draft step reads all of it
every single token. Restricting draft scoring to the 24,000 rows this model actually
uses cuts that to ~26 MB per step.

**The target model still verifies against the full vocabulary**, so the output
distribution is bit-for-bit unchanged. Only bytes are saved. Measured acceptance rate
did not drop.

**Why code and editing hit the high end:** both produce token streams where a short
suffix has appeared recently, which is exactly what an ngram drafter exploits — one
forward pass can carry a whole identifier, a `for` header, or a repeated phrase. In
editing mode, where most of the output is text already in the document, the drafter
predicts long spans and throughput climbs further. Prose has far less verbatim
repetition, so it lands lower. If your work is code or editing, the top of these ranges
is the realistic target.

---

## Get it running in three steps

Three pieces, deliberately kept in three places: weights here, the launcher and
configuration in a git repository people can read and patch, the engine binaries in a
release.

| What | Where |
|---|---|
| Weights (you are here) | this repo |
| Launcher `llmserver.bat` / `.sh`, **`features.ini`**, chat client | **[github.com/zgsndzq/qwen38-27b-flash-mtp](https://github.com/zgsndzq/qwen38-27b-flash-mtp)** |
| Prebuilt Windows runtime, CUDA DLLs included (**no CUDA Toolkit install**) | [Releases](https://github.com/zgsndzq/llama.cpp-qwen38-flash/releases) |
| Engine source and build scripts (only if you want to compile) | [llama.cpp-qwen38-flash](https://github.com/zgsndzq/llama.cpp-qwen38-flash) |

```bat
git clone https://github.com/zgsndzq/qwen38-27b-flash-mtp
cd qwen38-27b-flash-mtp
:: extract the release binary zip here, drop the two gguf files into weights\
llmserver.bat
```

### `features.ini` is not optional

It is **not** compiled into the binary — the engine has no built-in default path for
it and reads it at startup via `--features-file`. A binary built from source without
that file behaves like upstream, not like this package: you lose the 24k draft
shortlist, the tiered KV split, and probabilistic draft sampling.

Because the file is plain text and applied before argument parsing, every switch we
chose is inspectable and changeable without recompiling. One line in particular is
worth knowing about:

- `GGML_CUDA_FA_SHORTQ_MMA = 0` keeps the in-place quantised-KV attention kernel for
  short verify batches. Upstream #189 reroutes that shape onto an f16 MMA kernel,
  which on this model repeats a token and collapses draft acceptance. One line
  disables the detour.

### Reproduction settings

- Sampling: `--temp 0.6 --top-k 20 --top-p 0.95`
- KV cache `q4_0`; long context is made to fit by the tiered KV split (`kv-vram-cells`)
- `--spec-type ngram-mod,draft-mtp-adaptive` is **required** — without
  `draft-mtp-adaptive` the MTP head does not draft at all and throughput roughly halves
- `--spec-draft-vocab-map maps/shortlist_24k_mix.vocab.txt`
- Vision: add `--mmproj mmproj-Qwen3.8-27B-Q8_0.gguf --image-min-tokens 1024`

**This fork is mandatory.** `PTQ1_0` and the MTP draft context exist only here; stock
llama.cpp rejects the tensors. 8 GB cards can use `-c 65536`; on 6 GB drop to `32768`.

---

## Files

| File | Size | What it is |
|---|---|---|
| `Qwen3.8-27B-Flash-MTP.gguf` | 5.79 GB | **the model.** ternary backbone with the MTP draft head grafted in (866 tensors), embed/output layers kept at q4_K_M to preserve precision |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.59 GB | vision projector, only needed for image input |

sha256 values are in `SHA256SUMS`.

**Those two are all.** Other variants we tried during development (different MTP
donors, different embedding precision, and one build with no MTP head at all) are
deliberately not published, so nobody downloads a file that looks identical but
cannot self-speculate.

---

## Provenance and licence

**Licence: Apache 2.0**

This repository is a **derivative work**, not an original model. The full chain, every
layer under Apache 2.0:

| Layer | Source | What was done |
|---|---|---|
| 1 | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), Copyright 2026 Alibaba Cloud | the base model, 27B hybrid attention (~75% linear + 25% full) |
| 2 | [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), PrismML | matrix weights ternarised (PTQ1_0: ternary g128 + per-block Hadamard rotation + FP16 group scales) |
| 3 | [`ukisai/Swift-Bonsai-2-GGUF`](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF), UkisAI | reasoning-efficiency fine-tune (~39.8% fewer thinking tokens, benchmarks held) |
| 4 | MTP draft head, from `ProCreations` | recorded in this gguf as `graft.donor.name` |
| 5 | **this repository + [llama.cpp-qwen38-flash](https://github.com/zgsndzq/llama.cpp-qwen38-flash)** | merged layers 3 and 4 into one GGUF and repacked it at precision. **No retraining.** The speed work and the draft shortlist are this project's |

The engine (`zgsndzq/llama.cpp-qwen38-flash`, a fork of `PrismML-Eng/llama.cpp`) is
**MIT**.

**Required attribution:** *"Created using Bonsai by Prism ML."*

Using this repository means also complying with each layer's terms. See `NOTICE` and
`LICENSE`.

---

## Honest limitations

- **These are typical ranges, not guarantees.** Draft acceptance is
  workload-dependent: if your output has little verbatim repetition, expect the lower
  end. Open-ended prose generation lands noticeably below what code does on this
  hardware.
- **Cold-start first token is still 1–2 s.** The figures above describe steady state,
  not time-to-first-token.
- **Ternary quantisation costs agentic performance.** PrismML documents roughly 75% of
  the base model's SWE-bench Verified behaviour for the whole ternary family. Fast
  enough to be a daily driver for code, chat, document work and background jobs; not a
  replacement for a frontier model on long autonomous runs.
- **Prebuilt binaries are sm_89** (Ada). Ampere users must rebuild with
  `CUDA_ARCH=86`.
- **It is not an official Qwen release.** See the provenance table above — the base is
  Qwen3.8-27B and every modification is listed.

If your own measurement lands outside these ranges, open a discussion with your
`print_timing` line. A mismatch is more useful to us than a compliment.

---

## Citation

```bibtex
@misc{qwen38-27b-flash-mtp,
  title  = {Qwen3.8-27B-Flash-MTP: 27B ternary (PTQ1\_0) with MTP self-speculation, 8 GB VRAM},
  note   = {Derived from Qwen3.8-27B via PrismML Ternary Bonsai 2 and UkisAI Swift Bonsai 2},
  year   = {2026}
}
```

<details>
<summary><h3>中文说明</h3></summary>

## Qwen3.8-27B-Flash-MTP —— 27B 三值代码模型 + MTP 自投机，8 GB 显卡

**让普通用户进入 0 Token 时代。**

一台 **8 GB 显存的普通笔记本**，在家跑 27B 级代码模型。三值量化把 FP16 的 54 GB 压到
**5.79 GB**，MTP 自投机草稿头**内嵌在同一个 gguf 里**，写代码时保持交互速度。

| 场景 | 常见速度 |
|---|---|
| **代码生成** —— Python / C / SQL / Go / TS / Rust / JS | **约 70–90 t/s** |
| **编辑模式** —— 改已有文件或文档 | **约 100–200 t/s** |

测试环境：**RTX 4060 Laptop，8 GB**，关闭思考，用本项目的 fork 跑真实开发任务。
**这是常见区间，不是保证值**：实际落在哪，取决于你的输出里有多少内容能被 ngram 起草器
预测——这也是代码和编辑比自由写作快的原因。你自己测出来大概在这个范围里。

**88K 上下文，全在显存。** 用 `q3_0`（3-bit KV 缓存，每 token 14 KB，q4_0 是 18 KB），
同一张 8 GB 卡能把 88K token 的上下文整段留在显存。据我们所知，这是第一次让 27B 级模型
在消费级硬件上以这个速度跑完整的长上下文 agent。

同一类机器上，官方 4-bit 量化大约 **1–2 t/s**。

**为什么代码和编辑模式能跑到区间顶端**：这两种任务产出的 token 流里，常常有刚刚出现过的
片段，正好是 ngram 起草器最擅长的——一次前向就能带出一整个标识符、一行 `for` 头，或者一整段
重复短语。编辑模式下大部分输出本来就在文档里出现过，起草器能预测很长的跨度，吞吐更高。
自由写作几乎没有逐字重复，所以落到区间下沿。**如果你的工作是代码或编辑，区间顶端才是现实预期。**

### 三处下载

| 拿什么 | 在哪 |
|---|---|
| 权重 | 本仓库 |
| 启动脚本 `llmserver.bat` / `.sh`、**`features.ini`**、命令行客户端 | **[qwen38-27b-flash-mtp](https://github.com/zgsndzq/qwen38-27b-flash-mtp)** |
| 预编译 Windows 运行时（含 CUDA dll，**不用装 CUDA Toolkit**） | [Releases](https://github.com/zgsndzq/llama.cpp-qwen38-flash/releases) |
| 引擎源码 + 构建脚本 | [llama.cpp-qwen38-flash](https://github.com/zgsndzq/llama.cpp-qwen38-flash) |

⚠️ **`features.ini` 不是可有可无的**：它不编译进二进制，引擎没有内置默认路径，是启动时用
`--features-file` 读的。少了它，自己编出来的就是上游行为 —— 会丢掉 24k 起草短名单、
tiered KV 分箱、probabilistic 起草采样。其中一条值得单独说：`GGML_CUDA_FA_SHORTQ_MMA = 0`
保留原地量化 KV 内核，上游 #189 会把这种形状绕到 f16 MMA 上，在本模型上实测会重复 token、
接受率崩掉。

### 复现要点

- 采样 `--temp 0.6 --top-k 20 --top-p 0.95`
- KV 缓存 `q4_0`，长上下文靠 tiered KV（`kv-vram-cells`）分箱
- `--spec-type ngram-mod,draft-mtp-adaptive` **必须带**，否则 MTP 不参与起草，速度掉一半以上
- `--spec-draft-vocab-map maps/shortlist_24k_mix.vocab.txt`
- **必须用本 fork**：`PTQ1_0` 和 MTP 草稿上下文只在这里有，官方 llama.cpp 会直接拒绝加载
- 8 GB 卡 `-c 65536`；6 GB 降到 `32768`

### 说明与限制

- **区间不是保证值**：接受率强依赖工作负载，输出重复少就会落到区间下沿。自由长文写作会明显
  低于代码。
- 冷启动首 token 仍有 1–2 s，上面的数字是稳态。
- 三值量化在长链路 agentic 任务上有代价，PrismML 对整个三值家族的公开数据是 SWE-bench
  Verified 约保留基座模型的 75%。日常写代码、问答、文档处理够用，不适合替代前沿模型跑长任务。
- 预编译二进制是 sm_89（Ada），Ampere 需自行以 `CUDA_ARCH=86` 重编。
- **这不是 Qwen 官方发布**，底座是 Qwen3.8-27B，改动见上面的出处表。

### 出处与许可

**Apache 2.0**。本仓库是**衍生作品**，逐层链条（详见英文部分的 provenance 表）：

| 层 | 出处 | 做了什么 |
|---|---|---|
| 1 | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)　Copyright 2026 Alibaba Cloud | 原始模型 |
| 2 | [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)　PrismML | 矩阵权重转三值（PTQ1_0） |
| 3 | [`ukisai/Swift-Bonsai-2-GGUF`](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)　UkisAI | reasoning-efficiency 微调 |
| 4 | MTP 草稿头，来自 `ProCreations` | 见 gguf 的 `graft.donor.name` |
| 5 | **本仓库 + `llama.cpp-qwen38-flash`** | 合并第 3、4 层并按精度重新打包，**未重新训练** |

**必填署名：** *"Created using Bonsai by Prism ML."*

</details>