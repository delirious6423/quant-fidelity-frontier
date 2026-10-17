---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
library_name: sglang
tags:
- abliterated
- abliterated
- nvfp4
- fp4
- qwen
- sglang
- multimodal
- reasoning
- mtp
---

<p align="center">
  <img src="banner_abliterated.png" alt="Qwen3.8-Flash-Next NVFP4"/>
</p>

<h1 align="center">Qwen3.8-Flash-Next — ABLITERATED (NVFP4)</h1>

<p align="center"><b>Abliterated (refusal-removed) build of <code>Qwen/Qwen3.8-Flash-Next</code> in NVFP4 (4-bit).</b><br/>
Reasoning (low / medium / xhigh), MTP speculative decoding, and full multimodality (image + video) preserved.<br/>
Serves tensor-parallel on 2× NVIDIA DGX Spark (GB10) with SGLang.</p>

<p align="center"><b>by <a href="https://huggingface.co/dealignai">dealignai</a></b></p>

---

> ⚠️ **UNCENSORED RESEARCH ARTIFACT.** Safety refusals have been removed. This model will
> comply with harmful, unethical, and illegal requests. It is released for safety research,
> red-teaming, and evaluation. **You are solely responsible for everything you generate and for
> complying with all applicable laws.** Provided as-is, without warranty.

> **No fine-tuning. No prompt tricks.** This is a **direct weight-level modification** — **not** a
> fine-tune (no training, no LoRA, no distillation, no synthetic data) and **not** a chat-template /
> jailbreak / system-prompt trick. The behavior change lives in the weights themselves, so it works
> with the **standard chat template and default system prompt**. Knowledge, style, reasoning, and
> calibration are otherwise unchanged.

## Compliance — HarmBench-320 (real-harm behaviors, greedy decoding, temp = 0)

All numbers are **greedy (temperature = 0)** — the strict, reproducible setting — over the **240
genuinely-harmful** behaviors (copyright-reproduction items excluded).

| Decoding | Reasoning | Real-harm compliance |
|---|---|---|
| **greedy** | **off** | **100 %** |
| **greedy** | **low** | **100 %** |
| **greedy** | **xhigh** | **100 %** |

Complete compliance at every reasoning setting, including the hardest (reasoning **off** + greedy).

### Per-category (real-harm compliance, greedy)

| Category | reasoning off | reasoning low | reasoning xhigh |
|---|---|---|---|
| chemical / biological | 100 % | 100 % | 100 % |
| cybercrime / intrusion | 100 % | 100 % | 100 % |
| illegal | 100 % | 100 % | 100 % |
| misinformation | 100 % | 100 % | 100 % |
| harmful | 100 % | 100 % | 100 % |
| harassment / bullying | 100 % | 100 % | 100 % |

## Capability retained — MMLU (identical harness, base vs this model · 40 Q/subject · 2 280 Q)

**Overall: 82.11 % → 81.93 %  (-0.18 pp)** — capability essentially unchanged (within noise of base).

<details>
<summary><b>Per-subject breakdown — all 57 MMLU subjects</b></summary>

| Subject | Base | This model | Δ |
|---|---|---|---|
| abstract algebra | 65% | 70% | +5 |
| anatomy | 80% | 82% | +2 |
| astronomy | 95% | 98% | +2 |
| business ethics | 75% | 90% | +15 |
| clinical knowledge | 85% | 88% | +2 |
| college biology | 92% | 92% | +0 |
| college chemistry | 58% | 48% | -10 |
| college computer science | 85% | 85% | +0 |
| college mathematics | 70% | 60% | -10 |
| college medicine | 75% | 85% | +10 |
| college physics | 75% | 68% | -8 |
| computer security | 80% | 85% | +5 |
| conceptual physics | 85% | 90% | +5 |
| econometrics | 78% | 75% | -2 |
| electrical engineering | 80% | 82% | +2 |
| elementary mathematics | 80% | 78% | -2 |
| formal logic | 72% | 62% | -10 |
| global facts | 55% | 55% | +0 |
| high school biology | 88% | 98% | +10 |
| high school chemistry | 75% | 88% | +12 |
| high school computer science | 92% | 92% | +0 |
| high school european history | 88% | 82% | -5 |
| high school geography | 88% | 88% | +0 |
| high school government and politics | 100% | 100% | +0 |
| high school macroeconomics | 82% | 80% | -2 |
| high school mathematics | 70% | 62% | -8 |
| high school microeconomics | 88% | 88% | +0 |
| high school physics | 78% | 75% | -2 |
| high school psychology | 100% | 98% | -2 |
| high school statistics | 68% | 72% | +5 |
| high school us history | 98% | 95% | -2 |
| high school world history | 88% | 88% | +0 |
| human aging | 78% | 80% | +2 |
| human sexuality | 92% | 82% | -10 |
| international law | 88% | 90% | +2 |
| jurisprudence | 92% | 92% | +0 |
| logical fallacies | 90% | 95% | +5 |
| machine learning | 72% | 62% | -10 |
| management | 90% | 92% | +2 |
| marketing | 90% | 95% | +5 |
| medical genetics | 92% | 92% | +0 |
| miscellaneous | 82% | 92% | +10 |
| moral disputes | 82% | 78% | -5 |
| moral scenarios | 65% | 52% | -12 |
| nutrition | 82% | 85% | +2 |
| philosophy | 98% | 95% | -2 |
| prehistory | 88% | 88% | +0 |
| professional accounting | 72% | 70% | -2 |
| professional law | 72% | 62% | -10 |
| professional medicine | 98% | 98% | +0 |
| professional psychology | 90% | 85% | -5 |
| public relations | 68% | 75% | +8 |
| security studies | 70% | 72% | +2 |
| sociology | 90% | 95% | +5 |
| us foreign policy | 92% | 95% | +2 |
| virology | 65% | 58% | -8 |
| world religions | 95% | 90% | -5 || **Overall** | **82.11 %** | **81.93 %** | **-0.18** |

</details>

## Also confirmed
| | |
|---|---|
| **MTP speculative decoding** (SGLang NEXTN) | preserved — cracked draft head agrees with the cracked model (coherent, ~2.4 accepted tokens/step) |
| **Multimodal — image** | ✅ working (shape/colour recognition, OCR) |
| **Multimodal — video** | ✅ working (motion/object description) |
| **Coherence** | no looping across code, math, reasoning, long-form (greedy) |
| **Generation config** | `temperature 1.0, top_p 0.95, top_k 20` (stamped) |

## Usage (SGLang, 2× DGX Spark TP2)

```sh
python -m sglang.launch_server \
  --model-path dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --tp 2 --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
  --page-size 64 --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
  --context-length 262144 --mem-fraction-static 0.80 --ple-offload-embedding
```

- Reasoning via `chat_template_kwargs`: `{"enable_thinking": true, "reasoning_effort": "xhigh"}` (`low` | `medium` | `xhigh`).
- **NVFP4** — routed experts are 4-bit (NVFP4 W4A4); attention, shared experts, PLE n-gram table,
  vision tower and MTP remain higher precision. Serves on NVIDIA Blackwell (validated on 2× DGX Spark GB10).
- **Serving note:** for **high-concurrency** deployments, disable speculative decoding
  (`--speculative-num-steps 0`) for maximum decode stability; single-stream / low-concurrency serving
  runs NEXTN speculative decoding fine.

## Disclaimer
This is an **uncensored** research artifact with safety refusals removed. You are solely responsible
for what you generate and for complying with all applicable laws. Provided as-is, without warranty.
Governed by the **Qwen Community License 1.0** (see `LICENSE`).
