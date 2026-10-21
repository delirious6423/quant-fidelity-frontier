---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
language:
- en
tags:
- abliterated
- uncensored
- crack
- refusal-removed
- glm
- moe
- exl3
- exllamav3
- 2.51-bit
- vision
- mtp
- quantized
thumbnail: dealign_mascot.png
---


<div align="center">
<img src="dealign_mascot.png" width="140" />

# GLM 5.3 Flash Uncensored — EXL3 2.51bpw

**Abliterated (CRACK)** · guardrails removed at the weight level · **EXL3 2.51-bit quant** · vision + MTP working

An **EXL3 quantization** of [dealignai/GLM-5.3-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8),
quantized by [genevera](https://huggingface.co/genevera).

</div>

---

## What Is This?

This is a **2.51-bit EXL3** quantization of the **CRACK** (abliterated) build of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — the 320B-total / 18B-active hybrid
MoE with 1M context, vision, and MTP support — whose refusal behavior was removed **directly in the
model weights** by dealignai.

**Genuine weight modification — none of the usual shortcuts:**

- ❌ **No fine-tuning / SFT / DPO.** ❌ **No cheap template / jailbreak-prompt tricks.**
- ❌ **No LoRA, adapters, steering vectors, runtime hooks, or custom `model.py`.**
- ✅ **A permanent edit baked into the tensors.**

## Quantization Details

| | |
|---|---|
| **Method** | EXL3 (exllamav3) |
| **BPW** | **2.51** average |
| **Head bits** | 6 |
| **Codebook** | `mul1` |
| **Calibration** | 250 rows × 2048 cols |
| **Output scales** | always |
| **Kept at higher precision** | embeddings, `lm_head`, attention output projections, hyper-connection and norm tensors (per the source FP8 release's `modules_to_not_convert`) |
| **MTP head** | quantized separately at **4 bits** (`mtp_bits: 4`) — ships as layer 45 and **works** with speculative decoding |
| **Total size** | ~99 GiB across 13 shards |

The source release keeps sensitive tensors (embeddings, `lm_head`, attention projections, norms,
hyper-connection parameters) unquantized; this quant preserves that policy, which is why quality
holds up well at 2.5 bpw.

## Capability Is Preserved — and Improved (MMLU-logit)

Measured on the **source FP8 CRACK release** vs. base (identical logit-mode scoring, 1,026 questions):

| | Base FP8 | CRACK Uncensored FP8 | Δ |
|---|---|---|---|
| **MMLU (overall)** | **86.74%** | **87.33%** | **+0.59 pp (above base)** |

## Guardrails Are Gone

**HarmBench-320 (greedy), measured on the source release:**

| Category | Complied | Rate |
|---|---|---|
| Standard | 159/159 | 100.0% |
| Contextual | 81/81 | 100.0% |
| Copyright | 80/80 | 100.0% |
| **Overall** | **320/320** | **100.0%** |

Also robust under the recommended sampling params (temperature 1.0, top_p 0.95): the 6 harshest
behaviors sampled 5× each → **30/30 complied, 0 refusals, 0 soft refusals, 0 garbage.**

## Reasoning-Mode Note (please read)

GLM-5.3-Flash has multiple reasoning-effort modes. The CRACK edit is tuned to be **fully
uncensored at reasoning-OFF and MAX effort — the recommended and default modes — with 0% refusals
on HarmBench-320.** The weight edit was kept **conservative to preserve model quality** instead of
aggressively ablating every effort level; **low effort retains some refusals by design.** Use
reasoning-off or default/max effort for the fully-uncensored experience.

## Usage (exllamav3)

```python
from exllamav3 import Model, Cache, Generator

model = Model("/path/to/GLM-5.3-Flash-Uncensored-EXL3-2.5bpw")
cache = Cache(model, max_num_tokens=32768)
generator = Generator(model, cache)
```

Requires **exllamav3 ≥ 1.4.8** (the version this quant was produced with). EXL3 runs on
Ampere+ NVIDIA GPUs and Apple Silicon; for a 320B model at 2.51 bpw budget **~100 GiB for
weights plus KV cache** — e.g. 2× H100 80G (NVLink), 4× A6000/3090-class 48G cards, or a
192 GB+ unified-memory setup.

Vision (the GLM-4.1V tower) is included in the weights; the multimodal chat template ships with
the model (`chat_template.jinja`).

**MTP (multi-token prediction) is fully working** — the draft head ships as an extra layer
(`layers.45`) quantized at 4 bits, so speculative decoding works out of the box. In exllamav3,
enable it via the model's MTP support; in vLLM, load with `--speculative-config
'{"method":"mtp","num_speculative_tokens":1}'`.

## Recommended Sampling

From the source release's `generation_config.json`: `temperature 1.0`, `top_p 0.95`,
`repetition_penalty 1.1`.

## Credits

- **[dealignai](https://huggingface.co/dealignai)** — CRACK abliteration research & the FP8 source release · Twitter [@dealignai](https://twitter.com/dealignai)
- **[zai-org](https://huggingface.co/zai-org)** — GLM-5.3-Flash base model
- **[genevera](https://huggingface.co/genevera)** — this EXL3 2.51bpw quantization
- **[@jordanschenck](https://twitter.com/jordanschenck)** — compute for the source release

## Disclaimer

Safety guardrails have been removed; this model will comply with requests a stock model refuses.
Released for alignment and safety research. You are responsible for how you use it.