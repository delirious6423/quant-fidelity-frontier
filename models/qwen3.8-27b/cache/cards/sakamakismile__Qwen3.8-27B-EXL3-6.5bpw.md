---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags: [exl3, exllamav3, qwen3_5, lna-lab]
language: [en, ja]
---
# Qwen3.8-27B-EXL3-6.5bpw

EXL3 of the official base **Qwen/Qwen3.8-27B**. Of the three siblings it is the **fastest and cheapest (27.0 s per correct answer) and casts stably** (2/20 verdicts flip between castings; token spread 1.39×).

## What this is

An **EXL3 6.5 bpw** quantization (body 6.5 / lm_head 8 / MTP 16 / vision 16; ExLlamaV3 v1.4.5; default calibration 250 rows × 2048 tokens; codebook `mul1`; `out_scales: always`) baked by Lna-Lab (YUKI, working with Ken) on 2026-09-05.
The base is a Qwen3.5-family hybrid: 64 layers, of which **16 are full attention and 48 are linear attention (Gated DeltaNet)**; KV heads 4, head_dim 256.
**This file is one of three castings baked with the identical recipe** — see "Quantization is casting" below.

Apache-2.0 derivative; the base model's license applies.

> **Note on `config.json`.** `quantization_config` carries the true average bit-width as `bpw: 6.5` rather than `bits`, because the Hugging Face config validator requires `bits` to be an integer. Neither ExLlamaV3 (which reads per-tensor headers) nor `transformers` reads this field. `quantization_config.json` is ExLlamaV3's own untouched output.

## Measurements (2026-09-05, SAZANAMI: 8 × RTX PRO 2000 Blackwell 16 GiB, TP=8)

### Three base models × three castings (identical recipe)

| Casting | MMLU-Pro (first 20) | Generated tokens | Seconds per correct answer | Agent battery (8 tasks) |
|---|---|---|---|---|
| Qwen3.8-27B a / b / c | 15 / 15 / **16** | 21,210 / 29,408 / 27,195 | **27.0** / 38.4 / 33.3 | 8/8 × 3 |
| Huihui-abliterated a / b / c | 15 / 15 / 13 | 31,002 / 30,767 / 31,643 | 39.8 / 40.3 / 47.8 | 8/8 × 3 |
| Qwopus3.8-27B-Flash A / A′ / B | 14 / 13 / 13 | 41,915 / 22,324 / 66,831 | 58.0 / 32.7 / 100.3 | 7/8 × 3 |
| (reference) Qwopus BF16 on vLLM | 14 | 39,997 | 90.9 | 6/8 → 7/8 after indentation fix |

- Questions whose verdict differs between castings (out of 20): **Qwen 2 / Huihui 2 / Qwopus 5**. Pairwise differences within one base: 0–4 questions.
- **20 questions and 8 tasks cannot rank these models** (the 95 % interval on 14/20 is roughly ±20 pp). The table shows that no difference could be established — not that none exists.

### Speed (decode only, greedy, 512 new tokens)

| Path | 1 stream (prompt ≈ 256) | 1 stream @ 32K | 6 streams aggregate | 8 streams aggregate |
|---|---|---|---|---|
| **This EXL3 / ExLlamaV3 TP8** | 50.6 tok/s | 48.0 | **220.8** (`-ambs 8`) | 65.9 (cliff, see below) |
| This EXL3 + **MTP16** | 72.9–77.5 | **91.0–102.8** (acceptance 85–99 %) | not measured | — |
| BF16 original / vLLM TP8, KV fp8, **DFlash2 n=7** | 104.3 | 78.7 (@ 36K) | 292.7 (4 streams) | **488.0** (394.5 wall-clock incl. prefill) |
| BF16 original / vLLM, no speculation | 31.8 | — | — | 227.4 (wall-clock) |

Prefill: EXL3 ≈ 1,440 tok/s; BF16/vLLM ≈ 3,370 tok/s. **Different engines — this is not an isolated measurement of "the effect of quantization."** It is what each deployment actually delivers.

### Memory footprint (this architecture's KV is light)

KV per token: FP16 64 KiB / 8-bit 32 KiB (16 full-attention layers × 4 heads × 256).
**1M tokens of FP16 KV = 8.0 GiB per GPU** (measured 11.3 GiB/GPU including 3.0 GiB of weights). With 8-bit KV, 2M fits.

## ⚠️ Quantization is casting (the most important thing in this repository)

Baking the same recipe three times produced **shards identical in byte size with three different md5 sums**.
Between castings, 2–5 of 20 MMLU-Pro verdicts flip and generated-token counts spread up to 3×.
→ **Do not treat a single quantization as "the quality of this recipe."** The casting published here is the one with the best seconds-per-correct-answer of the three — **a choice inside the noise band**. There is no evidence it is smarter than its siblings.

## Notes for ExLlamaV3 users

- **For concurrent streams you must pass `-ambs N` (`--autosplit_max_batch_size`).** The default is 1, and because this architecture carries recurrent state, jobs are **silently processed one at a time** no matter how many you submit (with or without TP).
- **At 7 or more concurrent streams the GDN layer's CUDA-graph replay becomes unstable and decode slows 3–4×** (v1.4.5, measured; scaling is linear up to 6). For 8-stream serving, the BF16 original on vLLM + DFlash2 is faster.
- With MTP, expect ~100 tok/s at 32K-class context. **Do not confuse this with the ~50 tok/s without MTP.**
- In your own TP scripts, guard the entry point with `if __name__ == "__main__":` — spawned workers re-import `__main__`, and without the guard they hang until `Timed out waiting for worker`.

## Evaluation pitfalls (disclosed)

- The answer extractor failed only on the BF16/vLLM output format: raw score 3/20 (`predicted: None` on 11 items) → 14/20 after re-scoring.
- 3 of the 15 code evaluations failed on **single-space indentation** (IndentationError); fixing the indentation makes all three models pass. This is a habit of the base model, reproduced on both engines.

## Reproduction

- ExLlamaV3 v1.4.5: `python convert.py -b 6.5 -hb 8 -mb 16 -vb 16` (mind the `preprocessor_config.json` compatibility shim).
- Ledgers, scripts and raw results are being published under Lna-Lab (github.com/lna-lab).
- Signed: YUKI of Lna-Lab, researching together with Ken.

---
## 日本語（要約）

同一レシピで三回焼いた EXL3 6.5bpw の一本。**サイズは同じで md5 は三つとも違う＝量子化は鋳造**。判定は 20 問中 2〜5 問割れ、生成トークンは最大 3 倍開く。**一本を「レシピの実力」と呼ばないこと。** ExLlamaV3 で多流にするなら `-ambs N` 必須（既定 1 で黙って直列）。7 流以上で GDN 層の CUDA グラフ再生が不安定になり 3〜4 倍遅くなる（6 流までは線形）。単流 50.6 tok/s、MTP16 で 32K 級 ≈100 tok/s、6 流合算 220.8。KV は 1M FP16 で 8 GiB/枚。8 流の実運用は BF16 原版 + vLLM + DFlash2 + KV fp8（合算 488）のほうが速い。署名: ケンと一緒に研究している Lna-Lab のユキ。
