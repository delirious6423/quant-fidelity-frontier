---
language:
  - en
  - ko
  - zh
license: apache-2.0
tags:
  - gguf
  - qwen
  - qwen3.8
  - quantized
  - batiai
  - korean
  - vision
  - multimodal
  - ollama
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: llama.cpp
---

# Qwen3.8-27B GGUF — Quantized by BatiAI

<p align="center">
  <a href="https://flow.bati.ai"><img src="https://img.shields.io/badge/BatiFlow-on--device%20AI-blue?style=for-the-badge&logo=apple" alt="BatiFlow"></a>
  <a href="https://ollama.com/batiai/qwen3.8-27b"><img src="https://img.shields.io/badge/ollama-batiai%2Fqwen3.8--27b-black?style=for-the-badge&logo=ollama" alt="Ollama"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.8-27B"><img src="https://img.shields.io/badge/source-Qwen%20official-orange?style=for-the-badge" alt="Qwen"></a>
  <a href="#-license--apache-20"><img src="https://img.shields.io/badge/license-Apache--2.0-green?style=for-the-badge" alt="Apache 2.0"></a>
</p>

> ### Qwen's newest 27B — on your Mac, in Korean, in one line.
> **Weights dropped 2026‑08‑13.** Quantized the same week, straight from the official release.
> Six sizes from **10 GB**, vision projector included, **Korean output actually tested** — and
> Ollama tags published alongside, which most GGUF drops skip.

```bash
ollama run batiai/qwen3.8-27b:iq4
```

---

## 🧠 What Qwen3.8-27B actually is

Qwen's newest generation, built on the Qwen3.5 architecture — and the first 27B they call
"deployment‑friendly" while keeping frontier behaviour. Straight from the official card:

| | |
|---|---|
| **Context** | **262,144 tokens native**, extensible to 1M |
| **Architecture** | Hybrid: `16 × (3 × Gated DeltaNet → FFN, then 1 × Gated Attention → FFN)` — linear attention for most layers, full attention every 4th. That's why it stays fast at long context. |
| **Vision** | **Native** image *and* video understanding — STEM diagrams, documents, hour‑scale video |
| **Thinking** | On by default, disable per request, depth tunable via `reasoning_effort` |
| **Agents** | Stronger autonomous planning and environment‑feedback handling |
| **Params** | 27B dense · hidden 5120 · 64 layers · vocab 248,320 |

<sub>Architecture/context figures from the [official model card](https://huggingface.co/Qwen/Qwen3.8-27B).
Qwen's published benchmark numbers are for the **full‑precision base model** — we don't restate them
here, because what we can honestly report is what *these quants* did on our machine (below).</sub>

## 📦 Pick your size

| Quant | Size | Unified memory you need | Why this one |
|-------|------|------------------------|--------------|
| `Q2_K_S` | **10.2 GB** | **24 GB** | smallest, and the **fastest** — 28 t/s on M4 Max |
| `IQ3_XXS` | **11.2 GB** | **24 GB** | imatrix‑calibrated; better quality than Q2 at a small speed cost |
| `Q3_K_M` | **13.3 GB** | 32 GB | ⚠️ **only if `IQ4_XS` doesn't fit** — see the speed note below |
| **`IQ4_XS`** ⭐ | **15.1 GB** | **32 GB** | **the one to pick — faster *and* better than Q3_K_M on Apple Silicon** |
| `Q4_K_M` | **16.5 GB** | 36 GB+ | high quality, essentially the same speed as IQ4_XS |
| `Q6_K` | **22.1 GB** | 48 GB+ | near‑lossless |
| `mmproj` (BF16) | **0.93 GB** | + ~1 GB | vision projector — pair with any of the above for image input |

> ### ⚠️ Q3_K_M is smaller *and* slower on Apple Silicon. Skip it if you can.
>
> Size ordering does not predict speed here. Measured on M4 Max (Metal, `-ngl 99`):
>
> | Quant | Size | Generation |
> |---|---|---|
> | `Q3_K_M` | 12.38 GiB | **19.2 t/s** ← smaller |
> | `IQ4_XS` | 14.04 GiB | **22.4 t/s** (+17%) |
> | `Q4_K_M` | 15.40 GiB | **22.7 t/s** (+18%) |
>
> **Why**: generation speed is capped by `memory bandwidth ÷ model size` — every token reads the
> whole model. Against that ceiling (M4 Max, 546 GB/s), the efficiency numbers show what's really
> going on:
>
> | Quant | Ceiling | Measured | Efficiency |
> |---|---|---|---|
> | Q2_K_S | 53.4 t/s | 28.2 | **53%** |
> | IQ3_XXS | 48.8 | 23.7 | **48%** |
> | Q3_K_M | 41.1 | 19.2 | **47%** |
> | IQ4_XS | 36.2 | 22.4 | **62%** |
> | Q4_K_M | 33.0 | 22.7 | **69%** |
>
> The low-bit tiers sit at ~50% of their ceiling while Q4_K_M reaches 69%. They aren't
> bandwidth-limited — they're **compute-limited on dequantization**. Q2_K/Q3_K/IQ3 unpack in more
> steps and that costs Metal ALU time. So shrinking the file past Q4 buys you memory, not speed.
>
> On CUDA the same quants land at 70–80% efficiency across the board (50.8 / 50.3 / 46.2 t/s —
> effectively tied), so **this is an Apple Silicon effect.** Linux/CUDA users can pick on size alone.
>
> **And it isn't specific to this model.** Going back through our own benchmark archive, the same
> Q3_K_M → Q4_K_M ordering shows up across four unrelated model families and two different chips:
>
> | Model | Chip | Q3_K_M | Q4_K_M | Gap |
> |---|---|---|---|---|
> | Granite 4.1 (8B dense) | M4 Max | 60.8 | **77.4** | **+27%** |
> | Qwen3.8‑27B (this model) | M4 Max | 19.2 | **22.7** | +18% |
> | Gemma 4 26B (MoE) | M4 Max | 78.0 | **87.6** | +12% |
> | Granite 4.1 (8B dense) | **M4 (mini)** | 12.8 | **14.3** | +11% |
> | Qwen3.6‑27B | M4 Max | 15.3 | **16.6** | +8% |
>
> Dense and MoE, 8B and 27B, Max and non‑Max — same direction every time. Treat it as a property of
> Metal's Q3_K kernel, not of any one checkpoint.
>
> <sub>Rows other than Qwen3.8‑27B were measured through Ollama (`bench.sh`); the Qwen3.8‑27B row is
> `llama-bench`. Absolute values aren't comparable across harnesses — the **ordering within each row** is
> the finding. Qwen3.8‑27B numbers: llama.cpp b9430, M4 Max 128 GB, idle machine. The reversal was
> re‑confirmed on b10450.</sub>
>
> Practical reading: on a Mac, **`Q4_K_M` is the most efficient kernel** and `IQ4_XS` is within noise
> of it — pick IQ4_XS to save 1.4 GB, Q4_K_M if you have the room.

> ### 🚨 16 GB Macs: this model does not fit. Please don't waste the download.
>
> We originally listed 16 GB targets here. **That was wrong and we're correcting it.**
> A 16 GB Mac has roughly 12 GB usable after macOS, and measurement on an M4 Max shows
> `q2` — the *smallest* quant — sitting at a **14 GB floor even with context squeezed to 4096**:
>
> | `num_ctx` | Memory (`q2`, measured) |
> |---|---|
> | 262144 (model default) | **32 GB** |
> | 32768 | 16 GB |
> | 8192 / 4096 | **14 GB** (floor) |
>
> The weights alone are 10 GB; KV cache and runtime overhead do the rest. On 16 GB you get
> swapping, partial CPU fallback, and a model that feels broken. For 16 GB machines we
> recommend [Qwen3.6‑35B‑A3B](https://huggingface.co/batiai/Qwen3.6-35B-A3B-GGUF) (MoE, 3B active)
> or a smaller dense model instead.
>
> **Context length is the biggest memory lever here** — this checkpoint's native context is 262,144,
> so the default allocation is enormous. Our Ollama tags ship a modest `num_ctx` default;
> with llama.cpp set `-c` explicitly rather than inheriting the model default.
>
> <sub>Measured on M4 Max 128 GB by the BatiFlow Mac team, Ollama 0.20.0.
> Numbers for 24/32/36/48 GB rows are extrapolated from that floor plus quant size — we'll replace
> them with direct measurements as they come in.</sub>

All quantized **from Qwen's official BF16 weights** with an **imatrix calibrated on
wikitext-2**, then BatiAI‑signed (`general.author: BatiAI`).

<sub>An earlier version of this card described the calibration corpus as a mix of code, English,
Korean and Chinese. We measured it: it was 0.51% Korean and 0.01% Chinese — effectively English
wikitext. The line was wrong and we corrected it rather than leave it. Models we publish from
2026‑08‑27 on use a corpus we measure and report (see the composition figures on those cards).</sub>

> **All six were rebuilt on 2026‑08‑15.** Our first build mishandled this checkpoint's MTP layer —
> two files wouldn't load and every Ollama tag failed to start. We found it while benchmarking,
> pulled the broken files, and republished all six with the fix. Current files are verified on both
> engines. What went wrong and the correct flags are in "Two things that will bite you" below.

---

## ✅ Verified — captured from this build

<!-- VERIFY_BLOCK -->
| Test | Output |
|------|--------|
| Math | `127+58` → **185** |
| Korean | 한국의 수도는 서울입니다. |
| Tool call | `{"tool":"get_weather","args":{"city":"부산"}}` |
| Code | valid `binary_search` implementation (see below) |

<details><summary>Full captured output (greedy, temp 0)</summary>

````
[산수]
185

[한국어]
한국의 수도는 서울입니다.
비빔밥은 밥에 각종 채소와 고추장을 넣고 비벼 먹는 한국 대표 음식입니다.
불고기는 달콤한 양념에 재운 고기를 구워 먹는 인기 있는 한국 요리입니다.
떡볶이는 떡에 매콤달콤한 양념을 버무려 먹는 대표적인 한국 길거리 음식입니다.

[툴콜]
{"tool":"get_weather","args":{"city":"부산"}}

[코딩]
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
````

</details>
<!-- /VERIFY_BLOCK -->

> **Why we print this.** A quantized model can look fine on English benchmarks and fall apart
> in Korean — we've measured models that repeat a single word 60 times. So we run Korean,
> tool‑calling and code through every build we publish, and paste what came out. Good or bad.

---

## 🚀 Usage

### Ollama (simplest)

```bash
ollama run batiai/qwen3.8-27b:iq4     # 15.1 GB — 32 GB, the default pick
ollama run batiai/qwen3.8-27b:q2      # 10.2 GB — 24 GB, the smallest that's realistic
ollama run batiai/qwen3.8-27b:q4      # 16.5 GB — 36 GB+, if you have the headroom
```

### llama.cpp

```bash
hf download batiai/Qwen3.8-27B-GGUF Qwen3.8-27B-IQ4_XS.gguf --local-dir ./q38

./llama-cli -m ./q38/Qwen3.8-27B-IQ4_XS.gguf -ngl 99 -c 8192 -n 700 \
  -p "이 코드를 리팩터링하고 이유를 설명해줘."

# long context (262K native) — raise -c, watch RAM
./llama-cli -m ./q38/Qwen3.8-27B-IQ4_XS.gguf -ngl 99 -c 131072 -n 700 -f long_document.txt
```

### Vision (image input)

```bash
hf download batiai/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf mmproj-Qwen3.8-27B-BF16.gguf --local-dir ./q38

./llama-mtmd-cli -m ./q38/Qwen3.8-27B-Q4_K_M.gguf --mmproj ./q38/mmproj-Qwen3.8-27B-BF16.gguf \
  --image photo.jpg -p "이 이미지를 설명해줘."
```

Runs on **mainline llama.cpp** (arch `qwen3_5`) — no custom fork needed.
Ollama: **0.20+ works** — verified on 0.20.0 (Apple Silicon, Metal) and 0.20.6 (Linux, CUDA).
No recent-version requirement, so the one-line install path stays open.

The Ollama tags advertise `completion`, `tools` and `thinking`, so tool calls come back as
structured `message.tool_calls` and the reasoning block arrives separately in `message.thinking` —
you don't have to strip `<think>` yourself:

```bash
curl http://localhost:11434/api/chat -d '{
  "model":"batiai/qwen3.8-27b:iq4","stream":false,
  "options":{"num_predict":2500},
  "messages":[{"role":"user","content":"What is the weather in Busan?"}],
  "tools":[{"type":"function","function":{"name":"get_weather",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```

Give it room: **`num_predict` 2000+** for tool calls. The model reasons before it calls, and a
tight budget cuts it off mid-thought so the call never arrives.

Our Ollama tags ship `num_ctx 16384` rather than this checkpoint's native 262,144. The native value
would have Ollama reserve tens of GB of KV cache on first run — the single most common reason a
model "doesn't fit" on a Mac. **This is worth 30% of your throughput**, not just memory: on M4 Max,
`q2` went from 13.3 t/s at the native context to **17.3 t/s** at 16384, with memory dropping
32 GB → 14 GB. Raise it with `/set parameter num_ctx <n>` when you actually need
long context, and budget memory accordingly.

---

## ⚠️ Two things that will bite you (we hit both)

### 1. Thinking mode eats your token budget

Thinking is **on by default**. With a short `-n`, generation stops **inside the reasoning block**
and you get nothing usable — it looks like the model failed, but it just never reached the answer.

```bash
# ❌ looks broken — stops mid-thinking
./llama-cli -m Qwen3.8-27B-IQ4_XS.gguf -p "127+58은?" -n 64

# ✅ give it room
./llama-cli -m Qwen3.8-27B-IQ4_XS.gguf -p "127+58은?" -n 400

# ✅ or turn thinking off for short answers
./llama-cli -m Qwen3.8-27B-Q3_K_M.gguf --chat-template-kwargs '{"enable_thinking":false}' -p "127+58은?" -n 64
```

Rule of thumb from our runs: **400+ tokens for one‑liners, 700+ for anything explanatory.**

### 2. `blk.64` is an MTP layer — and pruning it is a trap

This checkpoint ships a multi‑token‑prediction layer that imatrix doesn't cover, so low‑bit
quantization fails on it (`Missing importance matrix for tensor blk.64.attn_k.weight`).

`--prune-layers 64` makes that error go away and produces a file that **looks** fine — correct
GGUF magic, sensible size, uploads cleanly — and then **segfaults on load**. We shipped two quants
that way for a few hours before a benchmark run caught it. The tensor count silently drops from
866 to 851 while the rest of the metadata still describes the full model.

Override the type instead of removing the layer:

```bash
llama-quantize --imatrix imatrix.dat \
  --tensor-type "blk\.64\..*=Q5_K" \
  Qwen3.8-27B-BF16.gguf out.gguf Q2_K_S
```

Q5_K doesn't require imatrix data, so the layer survives at a size that costs nothing measurable.
Nothing to do on your side when using our files.

---

## ⚡ Speed (measured, not estimated)

`llama-bench`, RTX 6000 Ada (48 GB), all layers offloaded, `pp512` / `tg128`:

| Quant | Size | Prompt (pp512) | Generation (tg128) |
|-------|------|---------------|--------------------|
| **Q3_K_M** | 12.6 GiB | 2068 tok/s | **50.8 tok/s** |
| IQ4_XS | 14.3 GiB | — | _rebuilding_ |
| Q4_K_M | 15.7 GiB | 2128 tok/s | **46.2 tok/s** |
| Q6_K | 20.9 GiB | 1810 tok/s | **35.7 tok/s** |

Quant size costs you generation speed almost linearly here — Q6_K is 30% slower than Q3_K_M
for weights you may not need. Prompt processing is flat across quants because it's compute‑bound,
not memory‑bound.

**How much you offload matters far more than which quant you pick** (Q3_K_M, `tg64`):

| `-ngl` (layers on GPU) | Generation |
|---|---|
| 99 (all) | **48.6 tok/s** |
| 40 | 13.3 tok/s |
| 20 | 8.4 tok/s |
| 0 (CPU only) | 6.0 tok/s |

Dropping from full offload to two‑thirds costs **73%** of your speed. On a Mac this is the same
cliff: if the model doesn't fit in unified memory and macOS starts swapping, you land in the
13 tok/s band and it feels like a different model. Check `ollama ps` says `100% GPU`.

**Apple Silicon — M4 Max 128 GB, Metal, `llama-bench`, `-ngl 99`:**

| Quant | Size | Prompt (pp512) | Generation (tg128) |
|-------|------|---------------|--------------------|
| Q2_K_S | 9.53 GiB | 242.0 tok/s | **28.2 tok/s** |
| IQ3_XXS | 10.41 GiB | 246.4 tok/s | **23.7 tok/s** |
| Q3_K_M | 12.38 GiB | 226.0 tok/s | **19.2 tok/s** |
| IQ4_XS | 14.04 GiB | 242.2 tok/s | **22.4 tok/s** |
| Q4_K_M | 15.40 GiB | 239.9 tok/s | **22.7 tok/s** |

Same model through Ollama 0.20.0 (`q2`): 13.3 tok/s warm, 5.0 s cold load, 100% GPU.
Don't compare that to the table — different harness, different measurement.

### Want it meaningfully faster? Change the model, not the quant.

A 27B dense model must read ~15 GB per token. No quantization setting escapes that. On the *same*
M4 Max 128 GB we measured:

| Model | Type | Generation |
|---|---|---|
| Qwen3.6‑27B `q4` | Dense 27B | 16.6 t/s |
| **[Qwen3.6‑35B‑A3B](https://huggingface.co/batiai/Qwen3.6-35B-A3B-GGUF)** `iq4` | **MoE, 3B active** | **45.1 t/s** |

The *larger* model runs 2.7× faster because only 3B parameters are active per token. If throughput
matters more to you than peak quality, that's the answer — quant tuning can't close a gap like that.

Things that do help this model, in order: a **recent llama.cpp build** (the Gated DeltaNet kernels
are still improving), `-fa` at longer context, and KV cache quantization (`-ctk q8_0 -ctv q8_0`)
past ~16K. Speculative decoding isn't available — Qwen3.8 ships only 27B and 2.4T‑A95B, so there's
no small draft model sharing this 248,320‑token vocabulary.

**Mac numbers come from Mac owners.** We only publish speeds measured on the actual hardware,
so Apple Silicon rows stay empty until someone runs it. If you have a Mac and five minutes,
[the benchmark kit is here](https://huggingface.co/batiai/Qwen3.8-27B-GGUF/discussions) —
copy‑paste one terminal block, send the numbers, and **your machine gets credited in this table**.

| Machine | Quant | Cold start | Generation | Measured by |
|---------|-------|-----------|------------|-------------|
| M4 Max 128 GB | Q2_K_S | 5.0 s | 28.2 tok/s | BatiFlow Mac team |
| _(your Mac could be here)_ | | | | |


---

## ✨ What BatiAI did differently

| | |
|---|---|
| 🇰🇷 **Korean verified** | Every quant family we ship gets Korean generation + tool‑call JSON checked. Most quant repos never test a non‑English language. |
| 🦙 **Ollama tags** | Published alongside the HF files. Most GGUF drops are HF‑only. |
| 📉 **Down to 10 GB** | Q2/IQ3 tiers built with imatrix so smaller machines get a real option — and we say plainly when a machine is too small rather than selling the download. |
| 👁️ **Vision included** | `mmproj` extracted and published — not text‑only. |
| 🔏 **From the source** | Quantized from Qwen's official weights, never a re‑quant of someone else's GGUF. imatrix‑calibrated, BatiAI‑signed. |

---

## 📜 License — Apache 2.0

Fully permissive: commercial use, modification, redistribution.
Base model © Alibaba Qwen team; quantized weights redistributed under the same terms.

## 🔗 Related

- Previous generation: [batiai/Qwen3.6-27B-GGUF](https://huggingface.co/batiai/Qwen3.6-27B-GGUF) · [batiai/Qwen3.5-27B-GGUF](https://huggingface.co/batiai/Qwen3.5-27B-GGUF)
- MoE sibling: [batiai/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/batiai/Qwen3.6-35B-A3B-GGUF)
- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

---

**Who we are.** [BatiAI](https://bati.ai) builds **on‑device Korean AI**.
[BatiFlow](https://flow.bati.ai) runs chat, **speech‑to‑text** ([batisay](https://huggingface.co/batiai/batisay-ko-turbo)),
**document OCR** ([batisee](https://huggingface.co/batiai/batisee)) and **speaker diarization**
entirely on a Mac — no audio, no documents, no prompts leave the device.
Full line: **[huggingface.co/batiai](https://huggingface.co/batiai)**
