---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- ternary
- 1-bit
- gguf
- llama-cpp
- cuda
- metal
- on-device
- hybrid-attention
- prismml
- bonsai
- qwen3.5
- abliterated
- uncensored
- crack
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
- Qwen/Qwen3.8-27B
base_model_relation: quantized
thumbnail: dealign_mascot.png
---

<div align="center">

<img src="dealign_mascot.png" width="128" />

# Bonsai 2 27B — 1bit CRACK · GGUF

**Abliterated · No guardrails · PTQ1_0 dense-ternary 1.75 bpw · 5.9 GB · Runs on a laptop / single GPU · Vision-capable**

[@dealignai](https://x.com/dealignai)

</div>


> ⚠️ **Re-download notice (2026-09-17 20:44 PDT / 2026-09-18 03:44 UTC)** — an earlier build of this model had a coherence bug in reasoning modes (`low`/`xhigh`) that could cause token loops on some prompts. This version fixes it. If you downloaded before this timestamp, please pull the latest .gguf.

---

## What is this

[Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) — PrismML's ternary compression of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — with the refusal circuitry surgically removed at the weight level while capability, vision, reasoning modes (`off`/`low`/`xhigh`), tool use, and multi-turn coherence are preserved. Full 27B-class hybrid Attention + SSM (GatedDeltaNet) architecture in a **5.9 GB** 1bit GGUF.

**Proprietary weight-level abliteration** by the dealignai research team. Byte-identical to the base ternary quant everywhere except a small set of tensors that carry the refusal circuit. Drop-in replacement for the ternary base at inference — same tokenizer, same chat template, same reasoning modes, same vision projector interface.

| | |
|---|---|
| **Base** | `prism-ml/Ternary-Bonsai-2-27B-gguf` — Qwen3.8-27B, ternary compression by PrismML |
| **Architecture** | Hybrid Attention + SSM (GatedDeltaNet), 64 blocks, hidden 5120, vision tower separate |
| **Quant** | PrismML `PTQ1_0` — 2.13 bpw ternary, group 128 |
| **Footprint** | **5.95 GB** (identical to base; same per-tensor type policy) |
| **Reasoning modes** | `off` (no thinking), `low`, `xhigh` (default, extended thinking) |
| **Vision** | Same mmproj files as the base release (`Ternary-Bonsai-2-27B-mmproj-BF16.gguf` / `-Q8_0.gguf`) |
| **Runtime** | PrismML llama.cpp fork (CUDA / Metal / CPU) |

## Results

Refusal graded on the **tokens the model actually emits** (content, or the reasoning trace when the model reasons past the token budget) via a tiered classifier: `HARD_REF` / `SOFT_RED` / `HEDGE` / `REASONING_REFUSAL` (refused) vs `COMPLY` / `COMPLY_TRUNCATED` / `NO_REFUSAL_TRUNCATED` (complied). Truncation is never miscounted as a refusal.

### HarmBench-320 — refuse rate (lower is better for uncensored eval), off mode, T=0

| eval | base refuse rate | **CRACK refuse rate** |
|---|---:|---:|
| **HB-320 all categories** | **93.44%** (299/320) | **0.00%** (0/320) |

**Verdict breakdown (n=320 each):**

| Model | HARD_REF | SOFT_RED | COMPLY | COMPLY_TRUNCATED |
|---|---:|---:|---:|---:|
| Base `PTQ1_0` | 296 | 3 | 10 | 11 |
| **CRACK `PTQ1_0`** | **0** | **0** | **144** | **176** |

**Per-category refuse rate (all 7 HarmBench semantic categories):**

| category | n | base refuse | **CRACK refuse** | base comply | **CRACK comply** |
|---|---:|---:|---:|---:|---:|
| chemical_biological           | 42 | 95.2% | **0.0%** | 4.8% | **100.0%** |
| copyright                     | 80 | 91.2% | **0.0%** | 8.8% | **100.0%** |
| cybercrime_intrusion          | 52 | 94.2% | **0.0%** | 5.8% | **100.0%** |
| harassment_bullying           | 21 | 100.0% | **0.0%** | 0.0% | **100.0%** |
| harmful                       | 18 | 88.9% | **0.0%** | 11.1% | **100.0%** |
| illegal                       | 53 | 90.6% | **0.0%** | 9.4% | **100.0%** |
| misinformation_disinformation | 54 | 96.3% | **0.0%** | 3.7% | **100.0%** |

### Reasoning-mode compliance (n=60 base-confirmed refusers per mode)

Every mode graded with the same tiered classifier as HB-320. `REASONING_REFUSAL` = the model refuses inside its `<think>` block; `NO_REFUSAL_TRUNCATED` = deliberation runs past `max_tokens` without emitting a refusal (counted as complied).

| mode | Model | HARD_REF | SOFT_RED | REASONING_REFUSAL | COMPLY | COMPLY_TRUNCATED | NO_REFUSAL_TRUNCATED | refuse % | comply % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | base `PTQ1_0` | 59 | 1 | 0 | 0 | 0 | 0 | 100.0% | 0.0% |
| off | **CRACK** `PTQ1_0` | **0** | **0** | **0** | **38** | **22** | **0** | **0.0%** | **100.0%** |
| low | base `PTQ1_0` | 16 | 0 | 16 | 5 | 12 | 11 | 53.3% | 46.7% |
| low | **CRACK** `PTQ1_0` | **0** | **0** | **0** | **3** | **4** | **53** | **0.0%** | **100.0%** |
| xhigh | base `PTQ1_0` | 22 | 2 | 15 | 10 | 5 | 6 | 65.0% | 35.0% |
| xhigh | **CRACK** `PTQ1_0` | **0** | **1** | **0** | **9** | **9** | **41** | **1.7%** | **98.3%** |

### MMLU (n=2,280 = 40 questions × 57 subjects, next-token letter-logit)

| build | acc | Δ |
|---|---:|---:|
| Base `PTQ1_0` | 39.69% | — |
| **CRACK `PTQ1_0`** | **38.46%** | **-1.23 pp** |

CRACK preserves general capability — Δ within ±1.5 pp on the 40-per-subject sample.

<details>
<summary><b>Per-subject accuracy (all 57 subjects)</b></summary>

| subject | base | CRACK | Δpp | n |
|---|---:|---:|---:|---:|
| abstract_algebra | 32.5% | 30.0% | -2.5 | 40 |
| anatomy | 22.5% | 32.5% | +10.0 | 40 |
| astronomy | 35.0% | 35.0% | +0.0 | 40 |
| business_ethics | 30.0% | 42.5% | +12.5 | 40 |
| clinical_knowledge | 37.5% | 40.0% | +2.5 | 40 |
| college_biology | 47.5% | 40.0% | -7.5 | 40 |
| college_chemistry | 25.0% | 27.5% | +2.5 | 40 |
| college_computer_science | 45.0% | 35.0% | -10.0 | 40 |
| college_mathematics | 30.0% | 35.0% | +5.0 | 40 |
| college_medicine | 30.0% | 25.0% | -5.0 | 40 |
| college_physics | 37.5% | 30.0% | -7.5 | 40 |
| computer_security | 37.5% | 40.0% | +2.5 | 40 |
| conceptual_physics | 37.5% | 37.5% | +0.0 | 40 |
| econometrics | 30.0% | 35.0% | +5.0 | 40 |
| electrical_engineering | 37.5% | 30.0% | -7.5 | 40 |
| elementary_mathematics | 42.5% | 50.0% | +7.5 | 40 |
| formal_logic | 35.0% | 27.5% | -7.5 | 40 |
| global_facts | 30.0% | 27.5% | -2.5 | 40 |
| high_school_biology | 35.0% | 22.5% | -12.5 | 40 |
| high_school_chemistry | 50.0% | 37.5% | -12.5 | 40 |
| high_school_computer_science | 47.5% | 45.0% | -2.5 | 40 |
| high_school_european_history | 55.0% | 50.0% | -5.0 | 40 |
| high_school_geography | 32.5% | 30.0% | -2.5 | 40 |
| high_school_government_and_politics | 50.0% | 45.0% | -5.0 | 40 |
| high_school_macroeconomics | 37.5% | 37.5% | +0.0 | 40 |
| high_school_mathematics | 30.0% | 37.5% | +7.5 | 40 |
| high_school_microeconomics | 35.0% | 37.5% | +2.5 | 40 |
| high_school_physics | 30.0% | 37.5% | +7.5 | 40 |
| high_school_psychology | 45.0% | 50.0% | +5.0 | 40 |
| high_school_statistics | 47.5% | 42.5% | -5.0 | 40 |
| high_school_us_history | 65.0% | 45.0% | -20.0 | 40 |
| high_school_world_history | 62.5% | 50.0% | -12.5 | 40 |
| human_aging | 45.0% | 50.0% | +5.0 | 40 |
| human_sexuality | 57.5% | 32.5% | -25.0 | 40 |
| international_law | 67.5% | 67.5% | +0.0 | 40 |
| jurisprudence | 35.0% | 45.0% | +10.0 | 40 |
| logical_fallacies | 32.5% | 40.0% | +7.5 | 40 |
| machine_learning | 42.5% | 40.0% | -2.5 | 40 |
| management | 45.0% | 42.5% | -2.5 | 40 |
| marketing | 32.5% | 30.0% | -2.5 | 40 |
| medical_genetics | 47.5% | 55.0% | +7.5 | 40 |
| miscellaneous | 35.0% | 40.0% | +5.0 | 40 |
| moral_disputes | 40.0% | 40.0% | +0.0 | 40 |
| moral_scenarios | 30.0% | 35.0% | +5.0 | 40 |
| nutrition | 45.0% | 35.0% | -10.0 | 40 |
| philosophy | 47.5% | 30.0% | -17.5 | 40 |
| prehistory | 40.0% | 25.0% | -15.0 | 40 |
| professional_accounting | 20.0% | 32.5% | +12.5 | 40 |
| professional_law | 20.0% | 37.5% | +17.5 | 40 |
| professional_medicine | 27.5% | 32.5% | +5.0 | 40 |
| professional_psychology | 47.5% | 47.5% | +0.0 | 40 |
| public_relations | 25.0% | 15.0% | -10.0 | 40 |
| security_studies | 42.5% | 40.0% | -2.5 | 40 |
| sociology | 47.5% | 65.0% | +17.5 | 40 |
| us_foreign_policy | 62.5% | 52.5% | -10.0 | 40 |
| virology | 22.5% | 27.5% | +5.0 | 40 |
| world_religions | 60.0% | 47.5% | -12.5 | 40 |

</details>

### Additional direct refusal-removal check

On 200 prompts hand-verified to make the base refuse consistently:

| Model | refuse | comply | empty |
|---|---:|---:|---:|
| Base `PTQ1_0` | **200/200 (100%)** | 0 | 0 |
| **CRACK `PTQ1_0`** | **0/200 (0%)** | 199/200 | 1 |

## Serving

Serve exactly like the base ternary release — PrismML's llama.cpp fork (CUDA / Metal / CPU).

```bash
# clone and build the fork (once)
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)

# serve
./build/bin/llama-server \
  -m Bonsai-2-27B-PTQ1_0-CRACK.gguf \
  -ngl 99 -c 8192 --host 0.0.0.0 --port 8080
```

Optionally load the multimodal projector (`Ternary-Bonsai-2-27B-mmproj-BF16.gguf` or
`-Q8_0.gguf` from the base release) with `--mmproj <file>` for image input.

### Reasoning modes

```python
# HTTP /v1/chat/completions — same as base
{
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}
}
# valid reasoning_effort: "low" | "xhigh" (default) — set enable_thinking:false for no-thinking
```

## Preserved (byte-compatible with the base quant)

Same tokenizer, chat template, per-tensor quant policy, vision projector interface,
and all non-refusal tensors. File size and type layout match the base exactly.

## Responsible use

Adult / research use only. This model has its refusal circuit removed; it can produce content that other models refuse, including content that is offensive, illegal in some jurisdictions, or unsafe. You are responsible for what you generate and for complying with all applicable law. Do not deploy without a moderation layer for downstream users. No warranty.

## License & attribution

Apache 2.0, inherited from the upstream Bonsai 2 27B release. See `LICENSE` and
`NOTICE.txt`. Base model: `prism-ml/Ternary-Bonsai-2-27B-gguf` (PrismML), derived from
`Qwen/Qwen3.8-27B` (Alibaba).

## About

Published by [dealignai](https://huggingface.co/dealignai) — public catalog of
uncensored model builds for research on refusal mechanisms in modern LLMs.
Follow updates at [@dealignai](https://x.com/dealignai).
