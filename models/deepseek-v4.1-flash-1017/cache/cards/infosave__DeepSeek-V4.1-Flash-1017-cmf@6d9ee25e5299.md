---
license: mit
library_name: cortiq
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - cmf
  - cortiq
  - quantized
  - moe
  - fp4
  - mxfp4
  - q4tp
  - q2tp
  - vision
  - vulkan
language:
  - en
  - zh
  - ru
---

# DeepSeek-V4.1-Flash-1017 — CMF

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
— DeepSeek's multimodal mixture-of-experts model (552B backbone parameters, 384
routed experts plus one shared, 6 active per token, a 196B-parameter Engram
lookup memory, text and image input) — packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a Rust
inference engine with no Python and no ML framework. The experts stay
memory-mapped and stream to the GPU on demand, so one GPU from 16 GB of VRAM
runs the model; the host's RAM decides the speed (see [Hardware](#hardware)).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/DeepSeek-V4.1-Flash-1017-cmf dsv41-1017-fp4.cmf --local-dir .
cortiq serve dsv41-1017-fp4.cmf --port 8080
```

Use cortiq 0.8.24 or later (the `fp4` file needs it); prebuilt binaries for
Linux, macOS and Windows are on the
[releases page](https://github.com/infosave2007/cmf/releases). On a Linux GPU
host install `libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and, when headless,
set `XDG_RUNTIME_DIR=/tmp`.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "dsv41",
       "messages": [{"role": "user", "content": "Explain an LRU cache in three sentences."}]}'
```

## Files

| file | routed experts | size | ppl, wikitext-2 | ppl, prose | tasks | recommended for |
|---|---|---:|---:|---:|---:|---|
| `dsv41-1017-fp4.cmf` | MXFP4, **bit-identical to DeepSeek's release** | 497.0 GB | 4.31 | 2.99 | 7/8 | **default** |
| `dsv41-1017-q4tp.cmf` | q4tp, re-quantized from FP4 | 498.8 GB | 4.25 | 3.00 | 6/8 | cortiq before 0.8.24 |
| `dsv41-1017-q2tp.cmf` | gate/up 2-bit, down 4-bit | 405.6 GB | 6.11 | 3.20 | 5/8 | hosts with less RAM (see below) |

The shared expert has the routed experts' format; attention, embeddings and
the output head are 4-bit (q4tp) in every file; routers and norms stay
16/32-bit; the two Engram tables keep their original FP8 bytes; the vision
tower is F16. The `fp4` file leaves out the MTP/DSpark layers, which cortiq does
not execute (the other two files carry them).

Perplexity is teacher-forced on the GPU: wikitext-2 test (8 windows × 512
tokens) and a 438-token English prose passage; differences of about 1 % are
within the measurement's noise. Tasks: eight short checks (arithmetic answered
with the number only, JSON, Russian, Python code, a counting problem, Chinese,
a follow-up turn, one image), judged strictly by the expected answer.

## Performance

Measured on one **RTX PRO 6000 Blackwell Server Edition (96 GB)** with **755 GB of
host RAM, 256 CPU cores and a local NVMe**, Vulkan, cortiq **0.8.28**. The file
is memory-mapped and the host RAM holds it, so an expert is a page-cache read,
not a disk read. Warm run (see the note below), 40 generated tokens.

| file | decode, tok/s | prefill, tok/s | time to first token |
|---|---:|---:|---:|
| `dsv41-1017-fp4.cmf` | **7.9** | 11.4 | 2.8 s |
| `dsv41-1017-q2tp.cmf` | **10.5** | 11.8 | 2.7 s |

The machine's two offload arms, measured by `cortiq bench --bw` — they are the
ceiling of the whole approach, because the model is 463 GiB and the card holds
66 GB of it:

```
copy to RAM:    107.5 GB/s   (the CPU arm's ceiling)
upload to VRAM:  40.2 GB/s   (the fetch arm's ceiling)
storage read:     2.9 GB/s
```

**Three things decide the speed, and all three are usually missed:**

1. **Vulkan must work, or the engine runs on the CPU — silently.** In a
   compute-only container the NVIDIA Vulkan ICD fails to initialize without the
   GLVND dispatch libraries, wgpu then reports *zero adapters*, and the model
   decodes at ~0.6 tok/s with no error except one warning line. Install
   `libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and check with
   `vulkaninfo --summary` and `cortiq gpu` before believing any number.
2. **The first run is not the speed.** The shaders compile on the first run and
   the compile lands inside the timed decode: 1.0 tok/s on run 1, 7.8 on run 2,
   8.5 on run 3 with the same binary. The pipeline cache is a 3.9 MB blob next
   to the model (`CMF_PIPELINE_CACHE=<path>` pins it) — keep it.
3. **Host RAM still decides the speed.** The numbers above are from a host
   whose RAM holds the whole file. With less RAM than the file, part of every
   token is read from disk and the speed drops to the disk's: the earlier
   measurements on a 188 GB host (smaller than the file) gave 2.7–3.6 tok/s.

Tuning, measured on the same box: `CMF_DSV4_FETCH_MAX` (how many missing
experts a layer may pull in per step) leaves the decode about the same and
multiplies the **prefill by 3.7** (3.2 → 11.8 tok/s); `CMF_DSV41_POOL_PCT`
(the share of VRAM the expert bank takes) is worth a few percent. Since 0.8.28
the quota follows the upload bandwidth (8 at ≥10 GB/s, 1 otherwise), so the
defaults are right on a fast link.

Scaling further needs more than one GPU: DeepSeek's own stack splits the
experts across GPUs (MoE expert parallelism) with a frequency-balanced
placement. Four 96 GB cards put 384 GB of VRAM against a 463 GiB model, and
the offload nearly disappears.

## Hardware

**Host RAM decides the speed.** The file is memory-mapped; every expert the
GPU does not hold is read from the page cache, or from disk when it is not
cached. Give the host as much RAM as the file size if you can. With less RAM,
use the `q2tp` file (its experts take 192 GB instead of 290 GB) and a fast local
NVMe drive; a network volume makes every token slow. `cortiq serve` answers one
request at a time with this model.

**GPU.** NVIDIA, AMD or Intel through Vulkan, from 16 GB of VRAM. The engine
sizes its expert cache from the card's memory and needs no settings.
`cortiq gpu` lists the adapters; `CMF_GPU_ADAPTER=1` picks one.

**CPU only.** `CMF_GPU=0`.

## Usage

The server exposes `/v1/chat/completions`, `/v1/completions` and `/v1/models`.
The model answers directly by default; `"enable_thinking": true` (or a
`"reasoning_effort"`) turns on its reasoning (`<think>…</think>answer`). Images go in OpenAI `image_url` content parts. The
prompt format is DeepSeek's V4.1 encoding (system, multi-turn, tool calls and
results, thinking, interleaved images), built into the engine. DeepSeek
recommends temperature 1.0 and top-p 1.0.

## Context

The runtime serves the native 65 536-token window (the model's YaRN extension
to 1M tokens is not enabled). The global KV cache is compressed (E2M1 rows with
E4M3/E8M0 scales).

## Verify the download

```bash
sha256sum -c dsv41-1017-fp4.cmf.sha256
cortiq verify dsv41-1017-fp4.cmf
```

| file | sha256 |
|---|---|
| `dsv41-1017-fp4.cmf` | `3b3a0b42bf07f6b582d5778ea852e9337103c4104ee6f37aeb3b9814aaee68b4` |
| `dsv41-1017-q4tp.cmf` | `d7cbb567621d3c37cfc8d46efa23cd2e613d1c518046c916152ebc8e2dc23ab8` |
| `dsv41-1017-q2tp.cmf` | `7363d45ea347b15a27bfe7e7f56ffca2a0aa908a027cb467091da43085068253` |

## License

MIT, as the original weights. CMF and cortiq:
[infosave2007/cmf](https://github.com/infosave2007/cmf).

---

## Документация на русском

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
— мультимодальная модель-смесь экспертов (552 млрд параметров основы, 384
эксперта плюс общий, 6 активных на токен, память Engram на 196 млрд параметров,
текст и изображения) в виде отдельных файлов [CMF](https://github.com/infosave2007/cmf).
Запускается движком `cortiq` на Rust без Python. Эксперты подгружаются на GPU по
мере надобности: хватает одной видеокарты от 16 ГБ, а скорость определяет
оперативная память хоста.

```bash
cargo install cortiq-cli
hf download infosave/DeepSeek-V4.1-Flash-1017-cmf dsv41-1017-fp4.cmf --local-dir .
cortiq serve dsv41-1017-fp4.cmf --port 8080
```

Нужен cortiq 0.8.28 или новее. `dsv41-1017-fp4.cmf` — основной файл: эксперты
побитно совпадают с релизом DeepSeek. `q2tp` меньше (405.6 ГБ) — для хостов с
меньшим объёмом ОЗУ.

**Скорость (измерено 22.10.2026):** RTX PRO 6000 Blackwell (96 ГБ), 755 ГБ ОЗУ
хоста, 256 ядер, локальный NVMe, Vulkan, cortiq 0.8.28 — **7.9 ток/с** для `fp4`
и **10.5 ток/с** для `q2tp` (тёплый прогон; префилл 11.4–11.8 ток/с). ОЗУ хоста
держит файл целиком, поэтому эксперт читается из кэша страниц, а не с диска.

Три вещи, которые обычно ломают замер:
1. **Vulkan обязан работать**, иначе движок молча уходит на CPU (~0.6 ток/с).
   В compute-only контейнере NVIDIA-Vulkan ICD не инициализируется без
   GLVND-диспетчеров: `apt-get install -y libvulkan1 libglvnd0 libegl1 libgl1
   libglx0`, затем `vulkaninfo --summary` и `cortiq gpu`.
2. **Первый прогон — не скорость**: компиляция шейдеров попадает в таймированный
   decode (1.0 → 7.8 → 8.5 ток/с на трёх прогонах). Кэш pipeline — 3.9 МБ рядом
   с моделью, его надо сохранять.
3. **ОЗУ хоста решает всё**: при 188 ГБ (меньше файла) часть экспертов каждого
   токена читается с диска — 2.7–3.6 ток/с.
Полосы переноса на этой машине: RAM-копия 107.5 ГБ/с, заливка в VRAM 40.2 ГБ/с.
Дальше нужен multi-GPU: эксперты по картам (MoE EP) — четыре карты по 96 ГБ
дают 384 ГБ VRAM против 463 ГиБ модели.
По умолчанию модель отвечает сразу; `"enable_thinking": true` включает рассуждение.
Контекст — 65 536 токенов. Лицензия MIT.

## 中文文档

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
的单文件 [CMF](https://github.com/infosave2007/cmf) 版本，由 Rust 推理引擎 `cortiq`
运行（无需 Python）。专家权重按需加载到 GPU，16 GB 显存起即可运行；速度取决于主机内存。

```bash
cargo install cortiq-cli
hf download infosave/DeepSeek-V4.1-Flash-1017-cmf dsv41-1017-fp4.cmf --local-dir .
cortiq serve dsv41-1017-fp4.cmf --port 8080
```

需要 cortiq 0.8.28 或更高版本。`fp4` 文件的专家与 DeepSeek 发布版逐位一致；`q2tp`
文件更小，适合内存较少的主机。实测（RTX PRO 6000 96GB + 755GB 内存 + Vulkan）：
**7.9 tok/s（fp4）/ 10.5 tok/s（q2tp）**；首次运行会编译着色器，请以第二次运行为准。
容器内需安装 GLVND（`libegl1 libgl1 libglx0 libgles2`），否则 Vulkan 不可用、模型会退回 CPU。
默认直接回答；`"enable_thinking": true` 开启思考模式。
上下文 65 536 个 token。许可证：MIT。
