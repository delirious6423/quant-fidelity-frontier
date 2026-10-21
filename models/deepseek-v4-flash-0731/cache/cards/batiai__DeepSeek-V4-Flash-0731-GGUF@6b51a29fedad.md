---
language:
  - en
  - ko
  - zh
license: mit
tags:
  - gguf
  - deepseek
  - quantized
  - batiai
  - mixture-of-experts
  - frontier
  - coding
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
library_name: llama.cpp
---

# DeepSeek-V4-Flash-0731 GGUF — Quantized by BatiAI

<p align="center">
  <a href="https://flow.bati.ai"><img src="https://img.shields.io/badge/BatiFlow-on--device%20AI-blue?style=for-the-badge&logo=apple" alt="BatiFlow"></a>
  <a href="https://github.com/batiai/bati.cpp"><img src="https://img.shields.io/badge/bati.cpp-runtime-black?style=for-the-badge&logo=github" alt="bati.cpp"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731"><img src="https://img.shields.io/badge/source-DeepSeek%20official-orange?style=for-the-badge" alt="deepseek"></a>
  <a href="#-license--mit"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT"></a>
</p>

> **The 2026-07-31 refresh of DeepSeek-V4-Flash — quantized from the official weights, verified in Korean.**
> 284B-A13B MoE with CSA+HCA hybrid attention. Runs on a single high-RAM Mac.

## 📦 Quantizations

| Quant | Size | Shards | Best for |
|-------|------|--------|----------|
| **Q3_K_M** | **135 GB** | 4 | M4 Max 128GB (tight) → 192GB Mac Studio |
| **Q4_K_M** | **172 GB** | 4 | M2 Ultra 192GB / server |

Built **directly from official DeepSeek weights** via a Q8_0 intermediate, BatiAI‑signed
(`general.author: BatiAI`). Q5+ intentionally omitted — the size/benefit tradeoff doesn't
land on any real Mac tier.

> 🆕 **What changed in 0731.** Same architecture as the original V4‑Flash (43 layers, 256 experts,
> vocab 129280) with refreshed weights — **and a new storage format**: experts are stored INT8 with
> **UE8M0 scales** (`scale_fmt: ue8m0`) instead of plain FP8. That is why these files are slightly
> larger than our earlier V4‑Flash GGUFs, and why conversion needs `torch>=2.7` (E8M0 dtype).

## ✅ Verified (this build, Q3_K_M)

Captured greedy runs — **not** copied from the base model card:

| Test | Output |
|------|--------|
| Math | `127+58=` → **185** |
| Korean | *"서울은 대한민국의 수도로, 역사와 현대가 공존하는 역동적인 도시입니다. / 불고기는 달콤한 간장 양념에 재운 소고기를 불에 구워내는… / 비빔밥은 밥 위에 다양한 나물과 고기, 고추장을 얹어… / 김치는 배추를 소금에 절여 고춧가루와 젓갈 등으로 양념해 발효시킨…"* — fluent, zero token mixing or loops |
| Tool call | `{"tool":"get_weather","args":{"city":"부산"}}` — exact JSON |

## 🚀 Usage

> ⚙️ **Runtime: [batiai/bati.cpp](https://github.com/batiai/bati.cpp)** — DeepSeek‑V4's CSA+HCA hybrid
> attention (`deepseek4`) is not in mainline llama.cpp. Build our fork.

```bash
hf download batiai/DeepSeek-V4-Flash-0731-GGUF "DeepSeek-V4-Flash-0731-Q3_K_M-*.gguf" --local-dir ./v4f

./llama-cli -m ./v4f/DeepSeek-V4-Flash-0731-Q3_K_M-00001-of-00004.gguf -ngl 99 -c 8192
```

### ⚠️ Two things the base repo doesn't tell you (we hit both)

1. **No chat template ships with this model** — neither the original nor the GGUF has one.
   Use DeepSeek's format explicitly:
   ```
   <｜User｜>your question here<｜Assistant｜>
   ```
2. **CUDA offload used to abort** in `ggml_cuda_op_concat` (`GGML_ASSERT(src0->type == GGML_TYPE_F32)`)
   because the backend advertised concat support for types its kernel can't handle.
   **Fixed in bati.cpp** — the CUDA backend now reports concat support only for F32, so those nodes
   fall back to CPU instead of crashing. Use bati.cpp at or after that fix, or run CPU/Metal.
   *(Apple Metal users were never affected.)*

## ✨ What BatiAI did

- Quantized **from the official DeepSeek weights** — never a re‑quant of someone else's GGUF
- Handled the new **UE8M0 / INT8‑expert** storage format end to end
- **Verified in Korean** — most GGUF publishers never check this
- **Fixed the runtime bug** we found along the way, in the open, in [bati.cpp](https://github.com/batiai/bati.cpp)
- BatiAI metadata‑signed

## 📜 License — MIT

Base model © DeepSeek, MIT. Quantized weights redistributed under the same terms.

---

**Who we are.** [BatiAI](https://bati.ai) builds **on‑device Korean AI** — [BatiFlow](https://flow.bati.ai)
runs LLMs, speech‑to‑text ([batisay](https://huggingface.co/batiai/batisay-ko-turbo)), document OCR
([batisee](https://huggingface.co/batiai/batisee)) and speaker diarization locally on a Mac.
No audio, no documents, no prompts leave the device.
Browse the whole line at **[huggingface.co/batiai](https://huggingface.co/batiai)**.
