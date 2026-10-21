---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - de
tags:
  - sakura
  - compact
  - gguf
  - qwen4exp
  - mixture-of-experts
  - strix-halo
  - vulkan
  - experimental
  - custom-runtime-required
---

# Sakura Qwen 3.8 Flash-Next Compact 47.6 GiB - K512 CP2 v1

![Sakura logo](assets/sakura-logo.svg)

> **47.56 GiB main model · 512 experts · Windows / Vulkan · MTP included**
>
> A smaller experimental community edition of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (Q2_0 quant). The two main-model files shrink from **61.86 to 47.56 GiB**, saving **14.30 GiB (23.1%)**, while retaining all **512 routed experts** and **16 lookup heads** of the unpruned base model. The matching custom runtime and materialized 512E MTP adapter are included separately.
>
> **This is the unpruned Qwen base model (NOT Swift, NOT REAP).**
>
> **Use the supplied runtime and START.cmd.** This experimental custom format requires its matching loader; generic GGUF launch instructions shown automatically by Hugging Face do not establish compatibility.

Made and tested on **AMD Strix Halo, Radeon 8060S, 64 GB UMA, Windows x64**, using Vulkan. Independent community release; not an official Qwen or ISTA-DASLab release.

## At a glance

| Item | This release |
|---|---|
| Model family | Qwen3.8-Flash-Next (base) |
| Architecture | `qwen4exp`, 48 layers |
| Routed experts | **512 experts retained (full, unpruned; 0 pruned)**; 10 active |
| Lookup heads | All 16 retained |
| Base quant source | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (Q2_0) |
| Main-model download | **47.56 GiB**, two GGUF shards |
| MTP adapter | **2.44 GiB** additional; included under `mtp/` |
| Main model + MTP | **50.00 GiB**, before runtime and other small files |
| Measured generation | **~33–37 token/s** with MTP2 / **38.9–43.8 token/s** in short checks |
| Included starter | One slot, 8K context capacity, Q8_0 KV, MTP2, GPU cache disabled |
| Status | Experimental; limited quality and context testing |

**Disk size is not running memory.** The tested 8K profile with MTP2 and Q8_0 KV is dimensioned for unified memory systems such as Strix Halo 64 GB UMA. A 47.6 GiB file size does not imply that this model fits a 32 GB or 24 GB GPU.

## Hardware compatibility

The status below applies to **this Compact CP2 v1 model and the supplied binaries**. Vulkan support in a GPU driver alone does not establish model or runtime compatibility.

| Hardware / platform | Status | What to expect |
|---|---|---|
| **AMD Strix Halo, Radeon 8060S, 64 GB UMA, Windows x64** | **Tested** | Supplied Vulkan runtime; one slot, 8K context capacity, Q8_0 KV and MTP2. Speed and test measurements are below. |
| Other Strix Halo configurations, including larger UMA capacities | **Not tested** | Similar hardware is not a separate validation. No measured speed, maximum context or memory guarantee for these configurations. |
| Other AMD Radeon GPUs on Windows | **Not tested** | The supplied backend is Vulkan. Driver features, host CPU instructions and available GPU/system memory must also match; no hardware-specific profile has been qualified. |
| NVIDIA GPUs on Windows | **Not tested** | No CUDA build is supplied. Vulkan operation with this custom format has not been validated on NVIDIA hardware. |
| Intel GPUs on Windows | **Not tested** | No tested Intel Vulkan profile or performance figures are available. |
| GPUs with 24 or 32 GB dedicated VRAM | **Default profile not validated** | The tested profile requires unified memory with sufficient headroom on UMA. A smaller-VRAM system may require different offloading and context settings; such profiles have not been tested. |
| CPU-only execution | **Not benchmarked or qualified** | The included launchers select Vulkan0. No CPU-only speed or RAM requirement is claimed. |
| Linux / macOS / Windows on ARM | **No matching native binary supplied** | This package contains Windows x64 binaries. Their native use on these platforms is not supported by the supplied package. |

### CPU, memory and storage

- **CPU:** the runtime was compiled as a native build on the test machine and tested on its AVX-512-capable x64 CPU. A portable AVX2-only build is not included. An AVX-512 label by itself is not a compatibility test for every other CPU.
- **Unified memory:** **64 GB UMA is the tested configuration**, not a measured minimum for every platform. Dedicated and shared Windows counters both draw from that same physical memory pool on Strix Halo. Leave room for Windows, CPU allocations and other applications.
- **Discrete GPUs:** separate system RAM does not provide the same behavior or bandwidth as Strix Halo UMA. Neither 48 GB nor a larger VRAM capacity alone guarantees compatibility; the device, driver, CPU and custom runtime all need validation.
- **Storage:** allow at least the **50.00 GiB model-plus-MTP footprint**, plus the runtime and supporting files. An SSD is recommended for the lazily read table. Free disk space does not substitute for sufficient runtime memory.

## Provenance and what changed

The model tree points directly to the common [Qwen base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Its actual lineage is:

1. [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
2. [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (Q2_0 quantization).
3. **This Compact CP2 v1 conversion.**

**Important distinctions:**
- **No Swift contribution:** Unlike the sister release `webmp3/Sakura-Qwen3.8-Flash-Next-Compact-43.9GiB-K352-CP2-v1-GGUF`, this release is based directly on the unpruned Qwen base quant.
- **512 experts:** All 512 routed experts are preserved. No expert pruning (REAP/Swift) was applied.
- **Tensor preservation:** All expert weights and all non-lookup tensor bytes are **100% byte-identical** to the source ISTA Q2_0 model. Only the large per-layer token lookup table was compressed using the CP2 lookup format, and associated GGUF metadata/filenames were updated.
- **Tokenizer preservation:** Target model's original tokenizer and chat template (`<|im_start|>`, Jinja template, etc.) are strictly preserved.
- **MTP adapter:** Materialized for this exact 512-expert model from the shared weights without retraining.

The package also includes a matching native runtime and a materialized MTP adapter for this exact 512E model. See [NOTICE.md](NOTICE.md) for the change notice and attribution.

## Files

| File or directory | Size / purpose |
|---|---|
| `Sakura-Qwen3.8-Flash-Next-K512-Q2_0-CP2-v1-Experimental-00001-of-00002.gguf` | **35.04 GiB** — load this first shard |
| `Sakura-Qwen3.8-Flash-Next-K512-Q2_0-CP2-v1-Experimental-00002-of-00002.gguf` | **12.52 GiB** — second shard, required alongside the first |
| `mtp/mtp-Qwen3.8-Flash-Next-K512-materialized-Agention.gguf` | **2.44 GiB** — matching MTP adapter for 512 experts |
| `runtime/` | Matching Windows x64 Vulkan binaries, DLLs and software license notices |
| `START.cmd` | Default launcher (calls `Start-Sakura-CP2.cmd`) |
| `Start-Sakura-CP2.cmd` | Server launch script configured for 8K context, MTP2, Vulkan |
| `results.json` | Public measurement summary and evaluation corpus hashes |
| `SHA256SUMS` | Release file checksums |
| `LICENSE-QWEN`, `NOTICE.md` | Model licenses, attribution and changes |

Keep both main shards in the same folder. Select shard 1; the runtime locates shard 2 automatically.

**Hugging Face hardware widget:** `Q2_0` in the filenames identifies the retained source expert-quantization tier; `CP2` identifies the custom lookup-table conversion. The widget may group this hybrid release under Q2_0. That label is not a certification of hardware fit or stock-loader compatibility: the included CP2 runtime is still required.

## Start on Windows

Download the package with its folder layout intact. Keep the model shards, launchers, `mtp/`, `runtime/` and license files together. No conversion is needed to use the supplied package.

1. Double-click **START.cmd** in the release folder.
2. Leave its terminal open while using the model.
3. Connect your OpenAI-compatible client to **http://127.0.0.1:8080/v1**. Query `/v1/models` for the model ID to use in the client.
4. Stop the server with **Ctrl+C** in its terminal.

The starter serves a **local API only** (`--no-webui`); it does not open a browser chat interface. It binds to `127.0.0.1`.

### Included profile

| Setting | Value |
|---|---|
| Backend | Vulkan, `Vulkan0` |
| GPU layers | `-ngl 99` |
| Context capacity | 8,192 tokens |
| Concurrent slots | 1 |
| Batch / microbatch | 512 / 512 |
| KV cache | Q8_0 for K and V |
| Flash attention | On |
| Table loading | `-lm dio --lazy-mode on` |
| Speculation | `draft-mtp`, maximum 2 draft tokens |
| Optional GPU row cache | 0 (disabled) |

### Manual equivalent

Run from the release folder in **Windows Command Prompt**:

```bat
set "SAKURA_CP2_GPU_CACHE_MIB=0"
runtime\llama-server.exe ^
  -m Sakura-Qwen3.8-Flash-Next-K512-Q2_0-CP2-v1-Experimental-00001-of-00002.gguf ^
  --host 127.0.0.1 --port 8080 --parallel 1 ^
  --device Vulkan0 -ngl 99 -c 8192 -b 512 -ub 512 ^
  -fa on -ctk q8_0 -ctv q8_0 -lm dio --lazy-mode on --jinja ^
  --spec-type draft-mtp ^
  --spec-draft-model mtp\mtp-Qwen3.8-Flash-Next-K512-materialized-Agention.gguf ^
  --spec-draft-ngl 99 --spec-draft-device Vulkan0 ^
  --spec-draft-n-max 2 --spec-draft-n-min 0 --no-webui
```

Use the provided 512E adapter with this model. Other shared MTP heads are not interchangeable.

## Measured quality: original ISTA Q2_0 versus Compact CP2

Own measurements on these exact conversion inputs and outputs. Lower perplexity is better.

| Fixed text slice | Original Q2_0 | Compact CP2 | Relative PPL change |
|---|---:|---:|---:|
| English (8 chunks) | 2.0115 | 2.0054 | -0.30% |
| Code (8 chunks) | 5.1468 | 5.1335 | -0.26% |
| German (8 chunks) | 2.1204 | 2.1421 | +1.02% |
| German (40 chunks) | 2.1289 | 2.1542 | +1.19% |

Each slice used 512-token chunks. Corpus identifiers are recorded in [results.json](results.json). In English and Code, perplexity remained virtually flat (-0.30% / -0.26%). In German, perplexity rose by +1.02% on 8 chunks and +1.19% on 40 chunks; these deviations are retained and openly documented. Per user priority, testing focused on English and code without further German optimization. These small changes do not establish general quality superiority.

## Generation speed and MTP speculation: own tests

**Hardware:** AMD Radeon 8060S, Strix Halo, 64 GB UMA, Windows, Vulkan. Tests dated **20 October 2026**.

- **Generation speed:** approximately **33–37 token/s** under sustained generation with MTP2; short 256-token smoke tests reached **38.92 / 43.78 token/s** (English) and **41.79 token/s** (Code).
- **MTP speculation acceptance:** **64.8–69.5%** with the supplied materialized 512E MTP adapter (`--spec-type draft-mtp --spec-draft-n-max 2`).
- **Answer parity:** Greedy short answer checks (e.g. Paris, 42) matched identically with and without MTP.

A broad standardized coding correctness evaluation has not been performed for this conversion. The model is an experimental release; no quality or correctness guarantee is claimed.

## Compatibility and known limits

- **Matching custom runtime required.** CP2 v1 is an experimental custom tensor format, not a stock GGUF quantization preset. Stock llama.cpp, Maya, LM Studio, Ollama and unrelated forks have not been qualified for this package.
- The included native binaries were tested on **Windows x64, an AVX-512-capable CPU and Radeon 8060S with Vulkan**. Other CPUs, GPUs, Linux and macOS are untested. Do not mix DLLs from unrelated runtime packages.
- The runtime derives from [Agention/llama.cpp](https://github.com/agentionai/llama.cpp) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The supplied build adds support for this Compact model. It has not been demonstrated to be the fastest available fork.
- Deep-context correctness above 8K, 128K operation, multiple concurrent slots, tool use and vision are unqualified for this conversion. No vision projector is included.
- Compact retains all 512 experts of the base model and adds lossy lookup-table compression.
- The supplied starter is a tested starting profile on one machine. Model file size alone is not a system-memory or VRAM requirement.

## Licenses and credits

Model weights retain the **[Qwen Community License 1.0](LICENSE-QWEN)**. Read the original license text before deployment. The native software's MIT and third-party notices are included under [runtime/](runtime/); software licenses do not replace the model-weight licenses.

Thanks to **Qwen**, **ISTA-DASLab**, **Agention** and the **llama.cpp / GGML contributors** for the upstream model, quantization and runtime work. Keep the supplied license texts and [NOTICE.md](NOTICE.md) when redistributing.

---

<details>
<summary>中文简要说明 · 樱花 Compact 512E</summary>

这是基于 ISTA-DASLab GSQ-RCO Q2_0 的全 512 专家版 Sakura 实验性紧凑版本。主模型的两个 GGUF 分片从 **61.86 GiB** 缩小到 **47.56 GiB**，节省 **14.30 GiB（23.1%）**。保留未剪枝基础模型的全部 **512 个专家**和 **16 个查找头**；专家权重及其他非查找表张量字节 100% 相同。**非 Swift 模型，无 REAP 剪枝。** 查找表采用有损 CP2 压缩。

**请使用本仓库附带的运行时和 START.cmd。** 512E 专属 MTP 适配器另占 **2.44 GiB**。保持目录结构完整，启动后客户端连接 `http://127.0.0.1:8080/v1`；启动脚本只提供本地 API，没有网页聊天界面。

测试平台：Windows x64、支持 AVX-512 的 CPU、AMD Strix Halo Radeon 8060S、64 GB 统一内存、Vulkan。测试配置为 8K 上下文容量、单槽位、Q8_0 KV、MTP2，GPU 行缓存关闭。实测持续生成速度约 **33–37 token/s**，短测试中达 **38.9–43.8 token/s**。配套 512E MTP 适配器的投机接受率为 **64.8–69.5%**。贪婪简短答案在启用/禁用 MTP 下完全一致。磁盘大小不等于运行内存需求；其他平台未验证。

英语和代码困惑度基本持平（-0.30% / -0.26%），德语困惑度略有上升（8块 +1.02%，40块 +1.19%）。语料哈希见 results.json。模型处于实验状态，不提供通用质量保证。

完整来源、文件列表、测试条件和限制见上方英文说明。模型权重受附带的 Qwen 许可约束；运行时软件许可不能替代模型许可。这是独立社区版本，不是 Qwen 或 ISTA-DASLab 官方发布。

</details>
