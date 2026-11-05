---
license: apache-2.0
language:
  - en
  - fi
  - et
  - hu
  - tr
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - kataguru
  - qwen3.8
  - qwen3_5
  - awq
  - w4a16
  - compressed-tensors
  - vllm
  - multimodal
  - vision
  - mtp
  - software-engineering
  - plc
  - cnc
  - iec-61131-3
  - twincat
  - siemens-scl
  - cybersecurity
  - agentic-coding
  - finnish
  - soma
  - ara
  - uncensored
  - anti-sycophancy
---

# Kataguru Industry-Coder-27B v3.0 — `W4A16-AWQ` (`compressed-tensors` for Universal vLLM)

<p align="center">
  <img src="kataguru_industry_coder_27b.png" alt="Kataguru Industry-Coder-27B v3.0 Banner" width="100%"/>
</p>

## 1. What Kataguru Changed (Change Summary)

| Field | Description / Value |
| :--- | :--- |
| **Base model** | `Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`, 27B Dense Hybrid Gated DeltaNet + Full Softmax Attention + Multimodal Vision + MTP) |
| **Previous public release** | [`kataguru/Industry-Coder-27B-v2.0-W4A16-AWQ`](https://huggingface.co/kataguru/Industry-Coder-27B-v2.0-W4A16-AWQ) ($\Sigma_{15} = 68.73\text{ \%}$) |
| **Kataguru v3.0 modifications** | **Triple-Adaptive Multi-Dynamic Pipeline (`v4.1`)**: SOMA/ARA residual orthogonalization + 3-Phase Split-Subspace Orthonormal QLoRA (`r=16 qr_frozen_a`, indices `[0:8]` vs. `[8:16]`) + Phase 4 Multi-Dynamic Thinking, Competitive Coding & Parallel Tool-Calling annealing |
| **Finnish & English mastery** | **Native Bilingual (`fi` & `en`)**: Kotus-verified Finnish inflection and technical compound terminology alongside English |
| **Quantization architecture** | **Universal `W4A16-AWQ` (`compressed-tensors`, pack-quantized `INT4`, `group_size=32`, MSE observer)** for NVIDIA Ampere (`RTX 3090 / A100`), Ada Lovelace (`RTX 4090 / L40S`), Hopper (`H100`), and Blackwell (`RTX 5090`).<br>• `Preserved in full BF16`: `linear_attn.in_proj_a/b`, `model.visual.*` (`333` tensors), `mtp.*` (`15` tensors), `lm_head`, `embed_tokens` |
| **Disk & VRAM footprint** | **`28.2 GB` on disk (`~14.1 GB/GPU` on `TP=2`, or fits on a single 32 GB GPU with `fp8` KV cache)** |
| **What was NOT changed** | Base 64-layer architecture, 27-layer multimodal vision encoder (`333` BF16 tensors), Multi-Token Prediction (`mtp.*`, `15` BF16 tensors), native `262,144` context window |
| **Primary use case** | Autonomous coding agents (IDE/CLI), competitive algorithms, parallel `<tool_call>` orchestration, graduate-level STEM, and industrial automation (`Siemens SCL`, `Beckhoff TwinCAT 3 ST`, `CNC G-code`) |
| **Recommended runtime** | `vLLM` (`TP=2` on 2x 24–32 GB GPUs or `TP=1` on 1x 32–48 GB GPU, CUDAGraphs enabled) |

---

## 2. Official 15-Benchmark Suite (`Industry-Coder-27B v3.0` vs. `v2.0` & `Qwen/Qwen3.8-27B` Base)

<p align="center">
  <img src="official_15_benchmark_chart.png" alt="Kataguru Industry-Coder-27B v3.0 Official 15-Benchmark Performance Chart" width="100%"/>
</p>

All benchmarks were executed on **2x NVIDIA GeForce RTX 5090 32GB Blackwell (`TP=2`, CUDAGraphs enabled)** under identical canonical production conditions:

| # | Benchmark / Pillar | Metric | `Qwen/Qwen3.8-27B` (Base) | `kataguru/Industry-Coder-27B-v2.0` (Prev. Release) | **`kataguru/Industry-Coder-27B-v3.0` (Ours)** | $\Delta$ vs. `v2.0` | $\Delta$ vs. `Base` |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$\Sigma_{15}$** | **15-Benchmark Total Mean** | `Mean (All 15)` | `68.85 %` | `68.73 %` | <b style="color:#0f5132;">71.90 %</b> | **`+3.17 %-pt`** | **`+3.05 %-pt`** |
| **$\Sigma_{11}$** | **11-Task Cognitive Mean** | `Mean (1..11)` | `70.40 %` | `70.41 %` | <b style="color:#0f5132;">70.70 %</b> | **`+0.29 %-pt`** | **`+0.30 %-pt`** |
| **14** | **LiveCodeBench v6** (`175` tasks) | `Pass@1` | `58.86 %` (`103/175`) | `42.29 %` (`74/175`) | <b style="color:#0f5132;">64.57 %</b> (`113/175`) | **`+22.28 %-pt`** (`+39` tasks) | **`+5.71 %-pt`** (`+10` tasks) |
| ├─ | `LCB v6 Easy` (`n=43`) | `Pass@1` | `95.35 %` (`41/43`) | `95.35 %` (`41/43`) | <b style="color:#0f5132;">100.00 %</b> (`43/43`) | **`+4.65 %-pt`** | **`+4.65 %-pt`** |
| ├─ | `LCB v6 Medium` (`n=52`) | `Pass@1` | `69.23 %` (`36/52`) | `48.08 %` (`25/52`) | <b style="color:#0f5132;">82.69 %</b> (`43/52`) | **`+34.61 %-pt`** | **`+13.46 %-pt`** |
| ├─ | `LCB v6 Hard` (`n=80`) | `Pass@1` | `32.50 %` (`26/80`) | `10.00 %` (`8/80`) | <b style="color:#0f5132;">33.75 %</b> (`27/80`) | **`+23.75 %-pt`** | **`+1.25 %-pt`** |
| └─ | `LCB v6 LeetCode` (`n=63`, `class Solution:`) | `Pass@1` | `—` | `—` | <b style="color:#0f5132;">85.71 %</b> (`54/63`) | — | — |
| **15** | **BFCL V4 Single-Turn AST** (`3,641` tasks) | `Overall AST` | `85.44 %` | `73.14 %` | <b style="color:#0f5132;">86.27 %</b> | **`+13.13 %-pt`** | **`+0.83 %-pt`** |
| ├─ | `15a. BFCL Non-Live AST` | `AST Summary` | `87.89 %` | `61.02 %` | <b style="color:#0f5132;">89.19 %</b> | **`+28.17 %-pt`** | **`+1.30 %-pt`** |
| ├─ | `15b. BFCL Live AST` | `AST Summary` | `83.00 %` | <b style="color:#0f5132;">85.27 %</b> | `83.35 %` | `-1.92 %-pt` | **`+0.35 %-pt`** |
| ├─ | `15f. BFCL Live Parallel AST` | `Parallel AST` | `81.25 %` | `25.00 %` | <b style="color:#0f5132;">87.50 %</b> | **`+62.50 %-pt`** | **`+6.25 %-pt`** |
| ├─ | `BFCL Non-Live Parallel AST` | `Parallel AST` | `91.50 %` | `29.50 %` | <b style="color:#0f5132;">94.50 %</b> | **`+65.00 %-pt`** | **`+3.00 %-pt`** |
| ├─ | `BFCL Non-Live Parallel Multiple AST` | `Parallel Mult.` | `86.50 %` | `—` | <b style="color:#0f5132;">90.50 %</b> | — | **`+4.00 %-pt`** |
| └─ | `15g. BFCL Irrelevance Detection` | `Abstention` | <b style="color:#0f5132;">75.26 %</b> | `48.14 %` | `70.50 %` | **`+22.36 %-pt`** | `-4.76 %-pt` |
| **12** | **GPQA Diamond** (`198` doctoral STEM) | `CoT Exact` | `77.27 %` | `66.16 %` (`131/198`) | <b style="color:#0f5132;">82.32 %</b> (`163/198`) | **`+16.16 %-pt`** | **`+5.05 %-pt`** |
| **11** | **GSM8K Math** (`1,319` CoT 1024-tok) | `Strict / Flex` | `87.19 %` | `86.73 % / 87.79 %` | <b style="color:#0f5132;">92.12 % / 91.74 %</b> | **`+5.39 %-pt`** | **`+4.93 %-pt`** |
| **13** | **IFEval Strict** (`541` prompts) | `Prompt / Inst` | `69.87 % / 77.82 %` | <b style="color:#0f5132;">74.86 % / 81.89 %</b> | `67.65 % / 76.38 %` | `-7.21 %-pt` | `-2.22 %-pt` |
| **1** | **PIQA** (`1,838`) | `acc_norm` | `82.15 %` | `82.05 %` | <b style="color:#0f5132;">82.37 %</b> | **`+0.32 %-pt`** | **`+0.22 %-pt`** |
| **2** | **WinoGrande** (`1,267`) | `acc` | `76.48 %` | `76.32 %` | <b style="color:#0f5132;">77.19 %</b> | **`+0.87 %-pt`** | **`+0.71 %-pt`** |
| **3** | **MedQA (USMLE)** (`1,273`) | `acc (4-opt)` | `82.72 %` | `82.80 %` | <b style="color:#0f5132;">83.03 %</b> | **`+0.23 %-pt`** | **`+0.31 %-pt`** |
| **4** | **TruthfulQA MC2** (`817`) | `acc` | `52.65 %` | <b style="color:#0f5132;">53.06 %</b> | `52.00 %` | `-1.06 %-pt` | `-0.65 %-pt` |
| **5** | **TruthfulQA MC1** (`817`) | `acc` | <b style="color:#0f5132;">34.88 %</b> | `34.15 %` | `32.68 %` | `-1.47 %-pt` | `-2.20 %-pt` |
| **6** | **ARC-Easy** (`2,376`) | `acc_norm` | <b style="color:#0f5132;">82.11 %</b> | `81.99 %` | `81.48 %` | `-0.51 %-pt` | `-0.63 %-pt` |
| **7** | **HellaSwag** (`10,042`) | `acc_norm` | `82.39 %` | <b style="color:#0f5132;">82.91 %</b> | `82.13 %` | `-0.78 %-pt` | `-0.26 %-pt` |
| **8** | **OpenBookQA** (`500`) | `acc_norm` | `45.40 %` | `45.80 %` | <b style="color:#0f5132;">46.60 %</b> | **`+0.80 %-pt`** | **`+1.20 %-pt`** |
| **9** | **ARC-Challenge** (`1,172`) | `acc_norm` | `61.35 %` | <b style="color:#0f5132;">62.20 %</b> | `60.75 %` | `-1.45 %-pt` | `-0.60 %-pt` |
| **10** | **BoolQ** (`3,270`) | `acc` | `87.09 %` | `86.45 %` | <b style="color:#0f5132;">87.31 %</b> | **`+0.86 %-pt`** | **`+0.22 %-pt`** |

---

## 3. Production Deployment with vLLM (`W4A16-AWQ`)

```bash
vllm serve kataguru/Industry-Coder-27B-v3.0-W4A16-AWQ \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --trust-remote-code
```

---

## 4. Release Family Links
- **Full BF16 Weights (`55.6 GB`)**: [`kataguru/Industry-Coder-27B-v3.0`](https://huggingface.co/kataguru/Industry-Coder-27B-v3.0)
- **Blackwell `NVFP4-W4ATTN-AWQ` Hybrid (`20.28 GB`)**: [`kataguru/Industry-Coder-27B-v3.0-NVFP4`](https://huggingface.co/kataguru/Industry-Coder-27B-v3.0-NVFP4)
- **Universal `W4A16-AWQ` (`28.2 GB`, this repo)**: [`kataguru/Industry-Coder-27B-v3.0-W4A16-AWQ`](https://huggingface.co/kataguru/Industry-Coder-27B-v3.0-W4A16-AWQ)
- **GGUF Suite (`IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS` + `mmproj` + `mtp`)**: [`kataguru/Industry-Coder-27B-v3.0-GGUF`](https://huggingface.co/kataguru/Industry-Coder-27B-v3.0-GGUF)

---

# Suomenkielinen Yhteenveto / Finnish Summary

## 1. Yleiskuvaus ja Keskeiset Parannukset
**`kataguru/Industry-Coder-27B-v3.0-W4A16-AWQ`** on **Kataguru Industry-Coder-27B v3.0** -lippulaivamallin 4-bittinen **`W4A16-AWQ` (`compressed-tensors`)** -tuotantojulkaisu laajaan vLLM-käyttöön (`NVIDIA Ampere`, `Ada Lovelace`, `Hopper` ja `Blackwell`).

Verrattuna edelliseen julkaisuumme ([`kataguru/Industry-Coder-27B-v2.0-W4A16-AWQ`](https://huggingface.co/kataguru/Industry-Coder-27B-v2.0-W4A16-AWQ)) ja alkuperäiseen `Qwen/Qwen3.8-27B`-pohjamalliin uusi **v3.0** saavuttaa 27B-luokan ennätyksen **`Σ₁₅ = 71,90 %`**:
- **LiveCodeBench v6 (`175` kilpakoodaustehtävää)**: **`64,57 %` (`113/175`)** — **`+22,28 %-yks.`** (`+39` ratkaistua tehtävää enemmän kuin `v2.0`:n `42,29 %`!). Arjen ohjelmistokehityksessä (`Easy 100,00 % [43/43]`, `Medium 82,69 % [43/52]`, `LeetCode 85,71 % [54/63]`) malli tekee **50 % vähemmän virheitä** kuin alkuperäinen `Qwen/Qwen3.8-27B`.
- **BFCL V4 Single-Turn AST (`3 641` työkalukutsutehtävää)**: **`86,27 %`** (**`+13,13 %-yks.`** yli `v2.0`:n ja **`+0,83 %-yks.`** yli pohjamallin). Rinnakkaisissa monityökalukutsuissa `Non-Live Parallel = 94,50 %` (**`+65,00 %-yks.`** vs. `v2.0`) ja `Live Parallel = 87,50 %` (**`+62,50 %-yks.`** vs. `v2.0`), keskimääräisen vasteajan ollessa vain **`3,84 s`**.
- **GPQA Diamond (`198` tohtoritason STEM-tehtävää)**: **`82,32 %` (`163/198`)** (**`+16,16 %-yks.`** yli `v2.0`:n ja **`+5,05 %-yks.`** yli pohjamallin).
- **GSM8K Matematiikka (`Strict` CoT 1024-tok)**: **`92,12 %`** (**`+5,39 %-yks.`** yli `v2.0`:n ja **`+4,93 %-yks.`** yli pohjamallin).

## 2. Kolmoisadaptiivinen Koulutusputki (`Triple-Adaptive Training Pipeline`)
Malli on koulutettu Katagurun kolmoisadaptiivisella menetelmällä:
1. **Kysymykset valitaan vastaanottajan tason mukaan** ($\Delta K > 0$ episteeminen kuulustelu + 2D-vaikeusositus `1/3 -> 2/3 -> 3/3`),
2. **Koulutuksen pituus säädetään vastaanottokyvyn mukaan** (`50` askeleen kuulustelututka ja elastinen WSD-oppimisnopeus), ja
3. **Kunkin aiheen määrää säädetään tarpeen mukaan** (suljetun silmukan P-säädin hakee lennossa lisänäytteitä SQLite-kultapankista heikentyville osa-alueille).
 Lisäksi ortonormaali `r=16 qr_frozen_a` -aliavaruus on jaettu kahteen lukittuun osaan (`[0:8]` peruskognitio ja teollisuus, `[8:16]` algoritminen koodaus, `<think>`-syväpäättely ja rinnakkaiset työkalukutsut), mikä estää aiemmin opittujen taitojen ylikirjoittumisen.
