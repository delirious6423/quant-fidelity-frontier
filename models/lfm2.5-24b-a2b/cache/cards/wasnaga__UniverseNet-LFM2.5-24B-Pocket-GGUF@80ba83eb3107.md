---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - lfm2
  - lfm2moe
  - liquid-ai
  - moe
  - llama-cpp
  - ios
  - mobile
  - on-device
  - tool-calling
  - agent
  - universe-network
---

# UniverseNet LFM2.5-24B: Pocket · Edge · Pro (GGUF)

**LiquidAI LFM2.5-24B-A2B, condensed with zero retraining into three llama.cpp checkpoints that run on hardware you already own, from an 8 GB iPhone to a 16 GB gaming GPU.**

<video src="https://huggingface.co/wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF/resolve/main/assets/demo_iphone_airplane.mp4" controls width="360"></video>

*Pocket v7.1 on an iPhone 16 Pro in airplane mode: real tool calls (device status, JavaScript, reminders, memory), unedited apart from captions.*

| | Pocket | Edge | Pro | Teacher |
|---|---|---|---|---|
| size | **3.996 GB** | **5.653 GB** | **7.31 GB** | 14.53 GB |
| iPhone 16 Pro, offline | **43.12 tok/s** | — | — | does not fit |
| RTX 4080 16 GB | 304.1 tok/s | 290.6 tok/s | 272.1 tok/s | 43.5 tok/s |
| public items right (of 106) | 50 | 76 | 97 | 96 |

![Hardware × accuracy: decode speed on five real devices](assets/hardware_x_accuracy.png)

- **No retraining, no data centre.** Each tier is produced by a closed-form condensation pass: no gradient steps, no fine-tuning afterwards. Pocket took 7.6–19.2 min, Pro 13.7 min, on one desktop RTX 4080 (16 GB).
- **Parameters removed, not just bits.** Every MoE expert keeps its job and the router is untouched; each expert's 1,536 feed-forward "orbitals" are condensed to 1,024 / 768 / 512. Attention, short-conv blocks, router and embeddings are bit-exact copies of the Teacher.
- **Stock runtimes.** Plain GGUF (Q3_K/Q4_K). Works in llama.cpp, LM Studio, Ollama and iOS apps such as PocketPal, with no custom kernels.


**Read more:** [the explainer (no physics degree needed)](https://huggingface.co/spaces/wasnaga/universe-network) · [the research paper](https://wasnaga-universe-network.static.hf.space/paper.html)

## Choose a tier

| File | Size | Params (total) | Orbitals / expert | Runs on | Public 6-suite score* |
|---|---|---|---|---|---|
| `LFM2.5-24B-A2B-Q4_K_M.gguf` (Teacher, reference) | 14.53 GB | 24.16 B | 1,536 | 24 GB+ Mac / partial CPU offload on 16 GB GPU | 95 / 95 / 100 |
| **`UniverseNet-LFM2.5-24B-Pro-16B-v6-Q3K.gguf`** | **7.31 GB** | 16.51 B | 1,024 | 12–16 GB GPU, 16 GB+ Mac | 95 / 95 / 100 |
| **`UniverseNet-LFM2.5-24B-Edge-12B-v6-Q3K.gguf`** | **5.653 GB** | 12.68 B | 768 | 8–12 GB GPU, 16 GB Mac / iPad | 90 / 80 / 90 |
| **`UniverseNet-LFM2.5-24B-Pocket-8.6B-v7.1-Q3K.gguf`** | **3.996 GB** | 8.86 B | 512 | **8 GB iPhone 16 Pro (offline)**, 8 GB Mac | 70 / 65 / 80 |

\*IFEval / GSM8K / BFCL-AST, % (full table below).

Earlier Pocket recipes live in `older/` for reproducibility: `v6` (previous default), `v7.0` (higher IFEval, weaker tool calling), and the first prototypes `v3` / `v4`. See *Pocket versions*.

## Files

```
UniverseNet-LFM2.5-24B-Pocket-8.6B-v7.1-Q3K.gguf   Pocket (default)
UniverseNet-LFM2.5-24B-Edge-12B-v6-Q3K.gguf        Edge
UniverseNet-LFM2.5-24B-Pro-16B-v6-Q3K.gguf         Pro
older/                                             Pocket v7.0, v6, v4, v3
evals/                                             raw result JSONs: public suites, Babbo journeys, iPhone benchmark, condensation logs
assets/                                            figures used on this card
BENCHMARKS.md · MANIFEST.md (sha256 of every GGUF)
```

## Hardware compatibility

✅ measured on that exact device · 🟢 fits by memory, not yet tested · ❌ does not fit

| Device (memory) | Pocket 4.0 GB | Edge 5.65 GB | Pro 7.31 GB | Teacher 14.5 GB |
|---|---|---|---|---|
| iPhone 16 Pro / 17 Pro (8 GB) | ✅ 43.12 tok/s | ❌ | ❌ | ❌ |
| iPhone 15 Pro, iPad (8 GB) | 🟢 | ❌ | ❌ | ❌ |
| iPhones with 6 GB or less | ❌ | ❌ | ❌ | ❌ |
| Mac mini / MacBook M1–M4 (8 GB) | ✅ 37.36 tok/s (M1) | ❌ | ❌ | ❌ |
| Mac with 16 GB (M1–M4), iPad Pro 16 GB | 🟢 | 🟢 | 🟢 | ❌ |
| MacBook Air M5 (24 GB) | ✅ 106.22 tok/s | ✅ 98.98 | ✅ 92.85 | ✅ 72.07 |
| 8 GB GPU (RTX 3060 Ti / 4060 / 3070) | 🟢 | 🟢 (short context) | ❌ | ❌ |
| 12 GB GPU (RTX 3060 / 4070) | 🟢 | 🟢 | 🟢 | ❌ |
| 16 GB GPU (RTX 4080 / 4070 Ti S / 5080) | ✅ 304.1 tok/s (4080) | ✅ 290.6 | ✅ 272.1 | partial CPU offload: 43.5 |
| 24 GB+ GPU (RTX 3090 / 4090 / 5090) | 🟢 | 🟢 | 🟢 | 🟢 |
| NVIDIA DGX Spark / GX10 (128 GB) | ✅ 174.8 tok/s | ✅ 163.5 | ✅ 151 | ✅ 119 |

Decode speed in tokens/s from `llama-bench` tg128, fully offloaded (`-ngl 99`). Leave about 1–2 GB above the file size for the KV cache and the OS. On iPhone, iOS limits a single app to roughly half of RAM, which is why only Pocket fits on 8 GB phones.

## Public benchmarks (same harness for Teacher and students)

Greedy, `llama-server`, `max_tokens 4096`, `fast` subsets: n=20 per suite (HumanEval 16, BFCL 10). Small samples: differences under ~15 points are not significant.

| Suite | Teacher | Pro | Edge | **Pocket v7.1** |
|---|---|---|---|---|
| IFEval (prompt strict) | 95 | 95 | 90 | 70 |
| MATH-500 | 65 | 75 | 45 | 20 |
| GSM8K | 95 | 95 | 80 | 65 |
| HumanEval pass@1 | 93.75 | 93.75 | 62.5 | 6.25 |
| ARC-Challenge | 100 | 95 | 70 | 50 |
| BFCL-AST (tool calls) | 100 | 100 | 90 | 80 |
| `</think>` closure / loop-free | 92.45 / 96.23 | 92.45 / 92.45 | 86.79 / 89.62 | 72.64 / 70.75 |
| held-out PPL (× Teacher 28.31) | 1.00 | 1.116 | 1.209 | 1.476 |

Pro matches the Teacher within noise on every suite at half the size. Edge keeps instructions and tools. Pocket keeps tool calling and instruction following but loses most multi-step arithmetic and code.

![Capability vs orbital budget](assets/lfm25_tiers.png)

## Speed on real devices

`llama-bench -p 512 -n 128 -r 5 -fa 1 -b 512 -ub 512 -ngl 99`, decode tokens/s. The iPhone runs the same loop in a minimal llama.cpp app, fully offline.

| Device | Teacher | Pro | Edge | Pocket |
|---|---|---|---|---|
| iPhone 16 Pro (8 GB) | does not fit | does not fit | does not fit | **43.12** (peak RSS 188.6 MB) |
| Mac mini M1 (8 GB) | does not fit | does not fit | does not fit | **37.36** |
| MacBook Air M5 (24 GB) | 72.07 | 92.85 | 98.98 | **106.22** |
| RTX 4080 (16 GB) | 43.5 (20 MoE layers on CPU) | 272.1 | 290.6 | **304.1** |
| NVIDIA GX10 (128 GB) | 119 | 151 | 163.5 | **174.8** |

Pocket rows for the iPhone, M1, M5 and GX10 were measured with the v6 file. v7.1 has identical tensor shapes (RTX 4080: v6 303 vs v7.1 304.1 tok/s). Where the Teacher fits in fast memory the condensed tiers are ~1.3–1.5× faster. Where it does not (16 GB GPU, 8 GB devices) they are 6–7× faster or the only option.

## Agentic test (Babbo family assistant, 11 multi-turn journeys)

Reminders with ISO-8601 dates across summer/winter time, shared memories, clarification turns, refusing unknown people.

| | Teacher | Pro | Edge | Pocket v7.1 |
|---|---|---|---|---|
| T=0, seed 42: summer / winter | 9/11 / 10/11 | 6/11 / 8/11 | 8/11 / 9/11 | 8/11 / 7/11 |
| T=0.3, 3 seeds: summer / winter (of 33) | 10/11 / 9/11 (1 seed, of 11) | 24/33 / 24/33 | 25/33 / 27/33 | 24/33 / 23/33 |

**Caveat:** the condensation calibration set includes traces from these journeys, so read this as a "seen-task" score, not general agent ability. Single-seed differences of 2–3 journeys are within seed noise.

## Pocket versions

| | v6 | v7.0 | **v7.1 (default)** |
|---|---|---|---|
| IFEval / MATH-500 / GSM8K | 60 / 10 / 60 | 80 / 15 / 55 | 70 / 20 / 65 |
| HumanEval / ARC-C / BFCL | 18.75 / 40 / 50 | 12.5 / 30 / 50 | 6.25 / 50 / 80 |
| Babbo T=0.3, 3 seeds (summer + winter, of 66) | 25/33 + 23/33 | 22/33 + 19/33 | 24/33 + 23/33 |
| held-out PPL | 41.9 | 41.62 | 41.79 |

v7 adds a "two-bond" constraint that keeps each condensed readout pointing where the Teacher's did (cosine 0.9555 vs 0.9467‡ for v6). v7.1 is the best Pocket on the public suites, especially tool calling.

## How it was made

Per layer, on the Teacher's own trajectory over a short calibration set: photoelectric + Pauli orbital selection, signed superposition of pruned twins into kept orbitals, a ridge "renormalization-group" fold of the pruned readout into the kept one, Cholesky error diffusion during Q3_K/Q4_K quantization, co-active Gauss–Seidel across the top-4 experts, and a per-layer gain clamp. Details, ablations and limitations: *Universe Network* paper (link TBD). Code: `exp12_lfm25_24b_pocket_moe_v5.py`.

![Pipeline](assets/condensation_pipeline.png)

## Run it

**iPhone / iPad / Mac:** in PocketPal AI (or LLM Farm), add from Hugging Face `wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF` and pick the Pocket file on 8 GB devices.

**llama.cpp:**
```bash
llama-server -m UniverseNet-LFM2.5-24B-Pocket-8.6B-v7.1-Q3K.gguf -ngl 99 -fa on -c 8192 --jinja --port 8080
```
Use `--jinja` for tool calling. Do **not** pass `--reasoning-format none`.

## Limitations

- Benchmarks are 10–20-item subsets per suite. Full splits are in progress.
- Pocket and Edge are weak at multi-step arithmetic and code. Use Pro for those.
- Babbo results are in-distribution for the calibration set.
- The source Teacher is already Q4_K_M; condensation re-quantizes dequantized weights.

## License

Derived from LiquidAI/LFM2.5-24B-A2B under the LFM Open License v1.0. See `license_link`.
