---
license: other
license_name: swift-open-license-v1.0
license_link: LICENSE
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

# Sakura Qwen 3.8 Flash-Next Compact 43.9 GiB - K352 CP2 v1

![Sakura logo](assets/sakura-logo.svg)

> **43.88 GiB main model · 352 experts · Windows / Vulkan · MTP included**
>
> A smaller experimental edition of [Sakura K352 IQ3](https://huggingface.co/webmp3/Sakura-Qwen3.8-Flash-Next-Swift-3bit-352E-GGUF), derived from UkisAI's Swift 1.5 fine-tune. The two main-model files shrink from **58.19 to 43.88 GiB**, saving **14.31 GiB (24.6%)**, while retaining all 352 experts of that Sakura source. The matching runtime and MTP adapter are included separately.
>
> **Use the supplied runtime and START.cmd.** This experimental custom format requires its matching loader; generic GGUF launch instructions shown automatically by Hugging Face do not establish compatibility.

Made and tested on **AMD Strix Halo, Radeon 8060S, 64 GB UMA, Windows x64**, using Vulkan. Independent community release; not an official Qwen or UkisAI release.

## At a glance

| Item | This release |
|---|---|
| Model family | Qwen3.8-Flash-Next, Swift 1.5 ancestry |
| Architecture | `qwen4exp`, 48 layers |
| Routed experts | 352 retained from the Sakura K352 input; 10 active |
| Lookup heads | All 16 retained |
| Main-model download | **43.88 GiB**, two GGUF shards |
| MTP adapter | **2.44 GiB** additional; included under `mtp/` |
| Main model + MTP | **46.32 GiB**, before runtime and other small files |
| Measured generation | **34.90 token/s** median in the short MTP2 test below |
| Included starter | One slot, 32K context capacity, Q8_0 KV, MTP2 |
| Status | Experimental; limited quality and context testing |

**Disk size is not running memory.** Windows reported about **39.79 GiB of GPU process memory** after warm-up with the default profile. Dedicated and shared allocations use the same physical UMA pool on this machine; they are not independent memory capacities. A 43.9 GiB file size does not imply that this package fits a 32 GB GPU.

## Hardware compatibility

The status below applies to **this Compact CP2 v1 model and the supplied binaries**. Vulkan support in a GPU driver alone does not establish model or runtime compatibility.

| Hardware / platform | Status | What to expect |
|---|---|---|
| **AMD Strix Halo, Radeon 8060S, 64 GB UMA, Windows x64** | **Tested** | Supplied Vulkan runtime; one slot, 32K context capacity, Q8_0 KV and MTP2. Short-test speed and memory measurements are below. |
| Other Strix Halo configurations, including larger UMA capacities | **Not tested** | Similar hardware is not a separate validation. No measured speed, maximum context or memory guarantee for these configurations. |
| Other AMD Radeon GPUs on Windows | **Not tested** | The supplied backend is Vulkan. Driver features, host CPU instructions and available GPU/system memory must also match; no hardware-specific profile has been qualified. |
| NVIDIA GPUs on Windows | **Not tested** | No CUDA build is supplied. Vulkan operation with this custom format has not been validated on NVIDIA hardware. |
| Intel GPUs on Windows | **Not tested** | No tested Intel Vulkan profile or performance figures are available. |
| GPUs with 24 or 32 GB dedicated VRAM | **Default profile not validated** | The tested profile used approximately 39.79 GiB of GPU process allocations on UMA. A smaller-VRAM system may require different offloading and context settings; such profiles have not been tested. |
| CPU-only execution | **Not benchmarked or qualified** | The included launchers select Vulkan0. No CPU-only speed or RAM requirement is claimed. |
| Linux / macOS / Windows on ARM | **No matching native binary supplied** | This package contains Windows x64 binaries. Their native use on these platforms is not supported by the supplied package. |

### CPU, memory and storage

- **CPU:** the runtime was compiled as a native build on the test machine and tested on its AVX-512-capable x64 CPU. A portable AVX2-only build is not included. An AVX-512 label by itself is not a compatibility test for every other CPU.
- **Unified memory:** **64 GB UMA is the tested configuration**, not a measured minimum for every platform. Dedicated and shared Windows counters both draw from that same physical memory pool on Strix Halo. Leave room for Windows, CPU allocations and other applications.
- **Discrete GPUs:** separate system RAM does not provide the same behavior or bandwidth as Strix Halo UMA. Neither 48 GB nor a larger VRAM capacity alone guarantees compatibility; the device, driver, CPU and custom runtime all need validation.
- **Storage:** allow at least the **46.32 GiB model-plus-MTP footprint**, plus the runtime and supporting files. An SSD is recommended for the lazily read table. Free disk space does not substitute for sufficient runtime memory.
- **Memory pressure:** use the supplied cache-off launcher as the first alternative to the default profile. Context or offload changes need their own correctness and performance checks. The optional cache is not a requirement to load the Compact model.

## Provenance and what changed

The model tree points directly to the common [Qwen base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Its actual ancestry remains:

1. [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
2. [UkisAI Swift 1.5 fine-tune](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next).
3. [UkisAI Swift GSQ-RCO GGUF quantization](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
4. [Sakura K352 IQ3](https://huggingface.co/webmp3/Sakura-Qwen3.8-Flash-Next-Swift-3bit-352E-GGUF), with 352 of the original 512 experts retained.
5. **This Compact CP2 v1 conversion.**

This conversion changes the representation of the large per-layer token lookup table and the associated GGUF metadata. **Expert weights and all other main-model tensor bytes are unchanged relative to Sakura K352.** No additional experts or lookup heads were removed, and no retraining was performed. The existing approximately 3-bit expert weights were retained; CP2 is not a new whole-model 2-bit quantization.

The package also includes a matching native runtime and a materialized MTP adapter for this exact K352 model. See [NOTICE.md](NOTICE.md) for the change notice and attribution.

## Files

| File or directory | Size / purpose |
|---|---|
| `Sakura-Qwen3.8-Flash-Next-Swift-K352-IQ3_XXS-CP2-v1-Experimental-00001-of-00002.gguf` | **20.01 GiB** — load this first shard |
| `Sakura-Qwen3.8-Flash-Next-Swift-K352-IQ3_XXS-CP2-v1-Experimental-00002-of-00002.gguf` | **23.87 GiB** — second shard, required alongside the first |
| `mtp/mtp-Sakura-K352-materialized-Agention.gguf` | **2.44 GiB** — matching MTP adapter |
| `runtime/` | Matching Windows x64 Vulkan binaries, DLLs and software license notices |
| `START.cmd` | Default launcher |
| `Start-Sakura-CP2-Cache-Off.cmd` | Alternative launcher with the optional GPU row cache disabled |
| `results.json` | Public measurement summary and evaluation corpus hashes |
| `SHA256SUMS` | Release file checksums |
| `LICENSE`, `LICENSE-QWEN`, `NOTICE.md` | Model licenses, attribution and changes |

Keep both main shards in the same folder. Select shard 1; the runtime locates shard 2 automatically. The long filenames retain `Swift` to reflect the input weights, even though the repository title uses the shorter Compact name.

**Hugging Face hardware widget:** `IQ3_XXS` in the filenames identifies the retained source expert-quantization tier; `CP2` identifies the custom lookup-table conversion. The widget may group this hybrid release under IQ3_XXS. That label is not a certification of hardware fit or stock-loader compatibility: the included CP2 runtime is still required. The measured memory figures above apply to the tested setup.

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
| Context capacity | 32,768 tokens |
| Concurrent slots | 1 |
| Batch / microbatch | 2048 / 2048 |
| KV cache | Q8_0 for K and V |
| Flash attention | On |
| Table loading | `-lm dio --lazy-mode on` |
| Speculation | `draft-mtp`, maximum 2 draft tokens |
| Optional GPU row cache | 1536 MiB in the default launcher |

If memory is tight, close other memory-heavy applications or use **Start-Sakura-CP2-Cache-Off.cmd**. The latter keeps the other profile settings and disables the optional 1536 MiB row cache. Reducing context capacity may also help; smaller custom profiles were not separately benchmarked here.

### Manual equivalent

Run from the release folder in **Windows Command Prompt**:

```bat
set "SAKURA_CP2_GPU_CACHE_MIB=1536"
runtime\llama-server.exe ^
  -m Sakura-Qwen3.8-Flash-Next-Swift-K352-IQ3_XXS-CP2-v1-Experimental-00001-of-00002.gguf ^
  --host 127.0.0.1 --port 8080 --parallel 1 ^
  --device Vulkan0 -ngl 99 -c 32768 -b 2048 -ub 2048 ^
  -fa on -ctk q8_0 -ctv q8_0 -lm dio --lazy-mode on --jinja ^
  --spec-type draft-mtp ^
  --spec-draft-model mtp\mtp-Sakura-K352-materialized-Agention.gguf ^
  --spec-draft-ngl 99 --spec-draft-device Vulkan0 ^
  --spec-draft-n-max 2 --spec-draft-n-min 0 --no-webui
```

Use the provided K352 adapter with this model. Other shared MTP heads are not interchangeable simply because they belong to the same model family.

## Measured quality: original Sakura K352 versus Compact

Own measurements on these exact conversion inputs and outputs. Lower perplexity is better.

| Fixed text slice | Original K352 | Compact CP2 | Relative PPL change |
|---|---:|---:|---:|
| English | 1.8636 | 1.8725 | +0.48% |
| Code | 5.0303 | 5.0758 | +0.90% |
| German | 2.0608 | 2.0766 | +0.77% |

Each slice used **eight 512-token chunks**. Corpus SHA-256 identifiers are recorded in [results.json](results.json). These are small fixed-text comparisons: they show a measurable compression trade-off, not a general guarantee of preserved reasoning or coding quality. Repeating the Compact PPL checks with the GPU row cache enabled produced the same rounded values.

Short greedy answer checks matched with and without MTP. **A broad coding correctness evaluation has not been completed for this Compact K352 build.** The parent repository's coding results concern the original model; they are not scores for this conversion. Results from the separate 512-expert base-model conversion also do not describe this Swift-derived K352 release.

## Speed and memory: own short tests

**Hardware:** AMD Radeon 8060S, Strix Halo, 64 GB UMA, Windows, Vulkan. Tests dated **20 October 2026**.

Same 1,728-token prompt, 200 generated tokens, one slot, 32K context capacity, batch/microbatch 2048, Q8_0 KV, temperature 0, seed 42 and MTP maximum 2. Two runs per setting after warm-up; medians below.

| Model / profile | Generation | Prompt processing | GPU process memory |
|---|---:|---:|---:|
| Original Sakura K352 IQ3 + MTP2 | 36.82 token/s | 417.9 token/s | 38.28 GiB |
| Compact CP2 + MTP2, row cache off | 33.82 token/s | 427.1 token/s | 38.28 GiB |
| Compact CP2 + MTP2, 1536 MiB row cache | **34.90 token/s** | 432.0 token/s | 39.79 GiB |

The original was measured with the preceding build before the optional GPU-cache extension. Compact cache-off/cache-on were measured directly with the same final build. Treat the original row as context rather than a fully identical-binary comparison.

Generation repeats were **31.37 / 36.28 token/s** with the cache off and **32.49 / 37.31 token/s** with it on. With this spread and only two repeats, **no reliable cache speed advantage is established**. The principal demonstrated benefit of this release is the smaller model download and disk footprint.

In an earlier direct Compact test, enabling MTP increased generation from **24.13 to 34.87 token/s (+44.5%)**, with identical greedy answers. That is an MTP benefit in that test, not evidence that table compression itself accelerates generation.

Memory figures are Windows process snapshots after warm-up, **not peak tracking**. The default-cache snapshot comprised about **28.37 GiB dedicated + 11.43 GiB shared**. Total adapter usage, including the desktop and other applications, was about **42.56 GiB**, with **3.43 GiB host memory free**. The remaining table is read lazily; Windows may cache file pages. The whole model is not held resident on the GPU.

The prompt occupied 1,728 tokens: **32K is configured capacity, not a demonstrated full-32K workload**. Other applications, longer prompts and concurrent requests can change both speed and memory use.

## Compatibility and known limits

- **Matching custom runtime required.** CP2 v1 is an experimental custom tensor format, not a stock GGUF quantization preset. Stock llama.cpp, Maya, LM Studio, Ollama and unrelated forks have not been qualified for this package.
- The included native binaries were tested on **Windows x64, an AVX-512-capable CPU and Radeon 8060S with Vulkan**. Other CPUs, GPUs, Linux and macOS are untested. Do not mix DLLs from unrelated runtime packages.
- The runtime derives from [Agention/llama.cpp](https://github.com/agentionai/llama.cpp) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The supplied build adds support for this Compact model. It has not been demonstrated to be the fastest available fork.
- Long/deep-context correctness, 128K operation, multiple concurrent slots, broad coding accuracy, tool use and vision are unqualified for this conversion. No vision projector is included.
- Compact retains the source's existing expert pruning and adds lossy lookup-table compression. Retaining all 352 source experts does not make it the original unpruned 512-expert model.
- The supplied starter is a tested starting profile on one machine. Model file size alone is not a system-memory or VRAM requirement.

## Licenses and credits

Model weights retain the **[Swift Open License v1.0](LICENSE)** and **[Qwen Community License 1.0](LICENSE-QWEN)** for their respective contributions. Read the original texts, including the Swift commercial-use provisions and Qwen service provisions, before deployment. The native software's MIT and third-party notices are included under [runtime/](runtime/); software licenses do not replace the model-weight licenses.

Thanks to **Qwen**, **UkisAI**, **ISTA-DASLab**, **Agention** and the **llama.cpp / GGML contributors** for the upstream model, fine-tune, quantization and runtime work. Keep the supplied license texts and [NOTICE.md](NOTICE.md) when redistributing.

---

<details>
<summary>中文简要说明 · 樱花 Compact</summary>

这是基于 UkisAI Swift 1.5 的 Sakura K352 实验性紧凑版本。主模型的两个 GGUF 分片从 **58.19 GiB** 缩小到 **43.88 GiB**，节省 **14.31 GiB（24.6%）**。保留源 Sakura 模型的全部 **352 个专家**和 **16 个查找头**；专家权重及其他主模型张量字节不变。没有新增剪枝或重新训练，查找表压缩是有损的。

**请使用本仓库附带的运行时和 START.cmd。** MTP 适配器另占 **2.44 GiB**。保持目录结构完整，启动后客户端连接 `http://127.0.0.1:8080/v1`；启动脚本只提供本地 API，没有网页聊天界面。

测试平台：Windows x64、支持 AVX-512 的 CPU、AMD Strix Halo Radeon 8060S、64 GB 统一内存、Vulkan。默认参数为 32K 上下文容量、单槽位、Q8_0 KV、MTP2 和 1536 MiB GPU 行缓存。短测试的生成速度中位数为 **34.90 token/s**，预热后的 GPU 进程内存约 **39.79 GiB**。磁盘大小不等于运行内存需求；其他平台未验证。

硬件兼容性：上述 8060S / 64 GB UMA 配置已经测试。其他 AMD、NVIDIA、Intel GPU 和其他 Strix Halo 配置均未验证；本包没有 CUDA、Linux、macOS 或 ARM 原生运行时，也未提供通用 AVX2 构建。24/32 GB 显存的独立显卡不在默认配置的验证范围内，独立显卡的系统内存不能直接等同于 Strix Halo UMA。建议使用 SSD，并预留操作系统和其他程序所需内存。

英语、代码、德语小规模困惑度测试分别变化 **+0.48%、+0.90%、+0.77%**，每类仅使用 8 个 512-token 块。没有证明缓存带来可靠加速；尚未完成本 Compact K352 的广泛编程正确率、深长上下文或多槽位测试。原版 Sakura 的编程分数不代表这个转换版本。

完整来源、文件列表、测试条件和限制见上方英文说明。模型权重仍受附带的 Swift 和 Qwen 许可约束；运行时软件许可不能替代模型许可。这是独立社区版本，不是 Qwen 或 UkisAI 官方发布。

</details>
