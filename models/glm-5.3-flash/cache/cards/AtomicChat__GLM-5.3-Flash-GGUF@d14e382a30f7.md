---
license: mit
license_link: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: AtomicChat
language:
- en
- zh
pipeline_tag: text-generation
library_name: gguf
tags:
- atomic-chat
- glm
- glm-5
- zai-org
- moe
- multimodal
- gguf
- imatrix
- quantized
- llama.cpp
---

# How to Run GLM-5.3-Flash Locally
<p style="margin-top: 0; margin-bottom: 0;">
  <em>Built from Z.ai's original weights with our own importance matrix. The <a href="https://huggingface.co/datasets/AtomicChat/calib-corpora">calibration corpora</a> behind our builds are public.</em>
</p>
<div style="display: flex; gap: 8px; align-items: center; margin-top: 10px; margin-bottom: 10px;">
  <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_glm_5_3_flash&utm_content=btn_atomic"><img src="https://huggingface.co/AtomicChat/GLM-5.3-Flash-GGUF/resolve/main/btn_atomic.png" width="162" alt="Atomic Chat"></a>
  <a href="https://discord.gg/8wGSsvmg4V"><img src="https://huggingface.co/AtomicChat/GLM-5.3-Flash-GGUF/resolve/main/btn_discord.png" width="119" alt="Discord"></a>
  <a href="https://github.com/AtomicBot-ai/Atomic-Chat"><img src="https://huggingface.co/AtomicChat/GLM-5.3-Flash-GGUF/resolve/main/btn_github.png" width="115" alt="GitHub"></a>
  <a href="https://x.com/atomic_chat_hq"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_x.png" width="108" alt="X"></a>
</div>
<ul style="margin: 0 0 12px 0;">
  <li>Built by the team behind <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_glm_5_3_flash&utm_content=bullet_app">Atomic Chat</a>, the app that runs open-weight models locally and offline.</li>
  <li>GLM-5.3-Flash is the first natively multimodal model in the GLM-5 series (320B total, 18B active).</li>
  <li>These GGUFs are self-quantized from Z.ai's original weights with our own importance matrix, published alongside the quants.</li>
  <li>The quants are still uploading and need a llama.cpp build with GLM-5.3-Flash support; Atomic Chat runs it as support ships.</li>
</ul>
<hr style="margin: 0 0 16px 0;">

<img src="https://huggingface.co/AtomicChat/GLM-5.3-Flash-GGUF/resolve/main/hero.png" alt="Z.ai" style="width:150px; max-width:100%; height:auto;"/>

## Highlights

- **320B total / 18B active** Mixture-of-Experts. The first natively multimodal model in the GLM-5 series; per Z.ai it outperforms GLM-5.2 across benchmarks at about one-tenth the price, and approaches Claude Opus 4.8 on coding and agentic tasks.
- **Hybrid attention**: combines sparse and linear attention to sharply cut long-context serving cost while preserving precise long-context capability. A first for the GLM series.
- **Manifold-Constrained Hyper-Connections (mHC)** to further improve scaling efficiency.
- **Newly trained base** on a 30T-token multimodal pre-training corpus.
- **Bilingual**, English and Chinese.
- **Natively multimodal** (text and vision). These GGUF quants cover the text path.
- **Frontier coding and agentic scores** (Z.ai-reported): Terminal-Bench 2.1 84.3, DeepSWE 1.1 63.4, HLE w/ tools 55.3, AutomationBench 48.8.
- **Full imatrix quantization** with our public [calibration corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora).

> [!NOTE]
> These GGUFs are **self-quantized from the original weights**, not a repack. The importance matrix keeps low-bit quants closer to the full-precision model.

> [!IMPORTANT]
> Always pass `--jinja` so the **GLM-5.3-Flash chat template** is applied. Without it the model can emit malformed turns.

## Model Overview

| Property | Value |
|---|---|
| Base model | `zai-org/GLM-5.3-Flash` |
| Total / active parameters | 320B total / 18B active |
| Architecture | Hybrid sparse + linear attention MoE with Manifold-Constrained Hyper-Connections (mHC) |
| Modality | Natively multimodal (text and vision); this repo covers the text path |
| Languages | English, Chinese |
| Pre-training | 30T-token multimodal corpus |
| Context length | Not stated by Z.ai; evaluations run up to 1,000,000 tokens with context management |
| This repo | GGUF quants (imatrix), text path. The importance matrix we built is published here too. |

<img src="https://huggingface.co/AtomicChat/GLM-5.3-Flash-GGUF/resolve/main/benchmark.png" alt="GLM-5.3-Flash benchmark scores" style="width:100%; max-width:900px;"/>

Scores are Z.ai's published results for the base `zai-org/GLM-5.3-Flash`. Quantization preserves the large majority of this; `Q4_K_M` and up sit within a point or two of full precision.


## Choosing a quant

| Quant | Size | Notes |
|---|---|---|
| `IQ2_M` | — | Smallest usable. Aggressive low-bit for memory-constrained boxes. |
| `IQ3_M` | — | Beats Q3 at similar size thanks to imatrix. Best low-RAM pick. |
| **`Q4_K_M`** | — | **Recommended default. Best balance of size, speed and quality.** |
| **`UD-Q4_K_XL`** | — | **Dynamic. Embeddings and output kept at Q8_0 for higher quality at a Q4 footprint.** |
| `Q6_K` | — | Near lossless. |
| `Q8_0` | — | Effectively lossless, reference quality. |

> [!TIP]
> Sizes fill in once the quants finish uploading. Pick the largest file that fits your (V)RAM with room for context.

## Get started

> [!NOTE]
> GLM-5.3-Flash uses a new hybrid sparse + linear attention architecture with Manifold-Constrained Hyper-Connections. The quants in this repo are still uploading, and running them needs a `llama.cpp` build that has landed GLM-5.3-Flash support. Until then, [Atomic Chat](https://atomic.chat) is the easiest way to run it as support ships.

Run GLM-5.3-Flash locally with:

- **[Atomic Chat](https://atomic.chat):** the easiest path. Open the app, search `AtomicChat/GLM-5.3-Flash-GGUF`, pick a quant, hit **Use this model**.
- **llama.cpp:** `llama-server -hf AtomicChat/GLM-5.3-Flash-GGUF:Q4_K_M --jinja -c 8192`
- **Ollama:** `ollama run hf.co/AtomicChat/GLM-5.3-Flash-GGUF:Q4_K_M`
- **LM Studio / Jan:** search the repo id, download any quant.

## Best practices

| Parameter | Value |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |

From Z.ai's evaluation settings (HLE w/ tools). Per-benchmark settings vary; see the base model card for details.

## Run in llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --target llama-cli llama-server
```

```bash
./llama.cpp/build/bin/llama-server \
    -hf AtomicChat/GLM-5.3-Flash-GGUF:UD-Q4_K_XL \
    --jinja -ngl 99 -c 8192 -fa on
```

## How these were made

1. Download `zai-org/GLM-5.3-Flash` (original weights).
2. Convert to GGUF with a [llama.cpp](https://github.com/ggerganov/llama.cpp) build that supports the GLM-5.3-Flash architecture (hybrid sparse + linear attention, mHC).
3. Build an importance matrix over our public [calibration corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora).
4. Quantize the ladder with `--imatrix`; `UD-Q4_K_XL` additionally pins the token-embedding and output tensors to `Q8_0`.

## License

Released by Z.ai (zai-org) under the MIT license. Quantized by Atomic Chat.

