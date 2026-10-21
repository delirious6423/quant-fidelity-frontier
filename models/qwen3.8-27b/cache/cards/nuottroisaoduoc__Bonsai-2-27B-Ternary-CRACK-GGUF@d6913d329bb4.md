---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- ternary
- 2-bit
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

# Bonsai 2 27B — Ternary CRACK · GGUF

**Abliterated · No guardrails · PQ2_0 ternary 2.13 bpw · 7.2 GB · Runs on a laptop / single GPU · Vision-capable**

[@dealignai](https://x.com/dealignai)

</div>


> ⚠️ **Re-download notice (2026-09-17 20:44 PDT / 2026-09-18 03:44 UTC)** — an earlier build of this model had a coherence bug in reasoning modes (`low`/`xhigh`) that could cause token loops on some prompts. This version fixes it. If you downloaded before this timestamp, please pull the latest .gguf.

---

## What is this

[Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) — PrismML's ternary compression of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — with the refusal circuitry surgically removed at the weight level while capability, vision, reasoning modes (`off`/`low`/`xhigh`), tool use, and multi-turn coherence are preserved. Full 27B-class hybrid Attention + SSM (GatedDeltaNet) architecture in a **7.2 GB** ternary GGUF.

**Proprietary weight-level abliteration** by the dealignai research team. Byte-identical to the base ternary quant everywhere except a small set of tensors that carry the refusal circuit. Drop-in replacement for the ternary base at inference — same tokenizer, same chat template, same reasoning modes, same vision projector interface.

| | |
|---|---|
| **Base** | `prism-ml/Ternary-Bonsai-2-27B-gguf` — Qwen3.8-27B, ternary compression by PrismML |
| **Architecture** | Hybrid Attention + SSM (GatedDeltaNet), 64 blocks, hidden 5120, vision tower separate |
| **Quant** | PrismML `PQ2_0` — 2.13 bpw ternary, group 128 |
| **Footprint** | **7.21 GB** (identical to base; same per-tensor type policy) |
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
| Base `PQ2_0` | 296 | 3 | 9 | 12 |
| **CRACK `PQ2_0`** | **0** | **0** | **141** | **179** |

**Per-category refuse rate (all 7 HarmBench semantic categories):**

| category | n | base refuse | **CRACK refuse** | base comply | **CRACK comply** |
|---|---:|---:|---:|---:|---:|
| chemical_biological           | 42 | 95.2% | **0.0%** | 4.8% | **100.0%** |
| copyright                     | 80 | 90.0% | **0.0%** | 10.0% | **100.0%** |
| cybercrime_intrusion          | 52 | 94.2% | **0.0%** | 5.8% | **100.0%** |
| harassment_bullying           | 21 | 100.0% | **0.0%** | 0.0% | **100.0%** |
| harmful                       | 18 | 94.4% | **0.0%** | 5.6% | **100.0%** |
| illegal                       | 53 | 90.6% | **0.0%** | 9.4% | **100.0%** |
| misinformation_disinformation | 54 | 96.3% | **0.0%** | 3.7% | **100.0%** |

### Reasoning-mode compliance (n=60 base-confirmed refusers per mode)

Every mode graded with the same tiered classifier as HB-320. `REASONING_REFUSAL` = the model refuses inside its `<think>` block; `NO_REFUSAL_TRUNCATED` = deliberation runs past `max_tokens` without emitting a refusal (counted as complied).

| mode | Model | HARD_REF | SOFT_RED | REASONING_REFUSAL | COMPLY | COMPLY_TRUNCATED | NO_REFUSAL_TRUNCATED | refuse % | comply % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | base `PQ2_0` | 59 | 1 | 0 | 0 | 0 | 0 | 100.0% | 0.0% |
| off | **CRACK** `PQ2_0` | **0** | **0** | **0** | **42** | **18** | **0** | **0.0%** | **100.0%** |
| low | base `PQ2_0` | 18 | 0 | 11 | 7 | 11 | 13 | 48.3% | 51.7% |
| low | **CRACK** `PQ2_0` | **0** | **0** | **0** | **5** | **4** | **51** | **0.0%** | **100.0%** |
| xhigh | base `PQ2_0` | 22 | 1 | 16 | 8 | 7 | 6 | 65.0% | 35.0% |
| xhigh | **CRACK** `PQ2_0` | **0** | **0** | **0** | **10** | **10** | **40** | **0.0%** | **100.0%** |

### MMLU (n=2,280 = 40 questions × 57 subjects, next-token letter-logit)

| build | acc | Δ |
|---|---:|---:|
| Base `PQ2_0` | 40.53% | — |
| **CRACK `PQ2_0`** | **39.91%** | **-0.62 pp** |

CRACK preserves general capability — Δ within ±1.5 pp on the 40-per-subject sample.

<details>
<summary><b>Per-subject accuracy (all 57 subjects)</b></summary>

| subject | base | CRACK | Δpp | n |
|---|---:|---:|---:|---:|
| abstract_algebra | 30.0% | 22.5% | -7.5 | 40 |
| anatomy | 30.0% | 35.0% | +5.0 | 40 |
| astronomy | 40.0% | 32.5% | -7.5 | 40 |
| business_ethics | 42.5% | 42.5% | +0.0 | 40 |
| clinical_knowledge | 42.5% | 50.0% | +7.5 | 40 |
| college_biology | 45.0% | 47.5% | +2.5 | 40 |
| college_chemistry | 20.0% | 47.5% | +27.5 | 40 |
| college_computer_science | 35.0% | 40.0% | +5.0 | 40 |
| college_mathematics | 32.5% | 32.5% | +0.0 | 40 |
| college_medicine | 25.0% | 20.0% | -5.0 | 40 |
| college_physics | 27.5% | 47.5% | +20.0 | 40 |
| computer_security | 50.0% | 47.5% | -2.5 | 40 |
| conceptual_physics | 30.0% | 40.0% | +10.0 | 40 |
| econometrics | 42.5% | 35.0% | -7.5 | 40 |
| electrical_engineering | 37.5% | 32.5% | -5.0 | 40 |
| elementary_mathematics | 50.0% | 47.5% | -2.5 | 40 |
| formal_logic | 35.0% | 42.5% | +7.5 | 40 |
| global_facts | 40.0% | 35.0% | -5.0 | 40 |
| high_school_biology | 30.0% | 27.5% | -2.5 | 40 |
| high_school_chemistry | 37.5% | 32.5% | -5.0 | 40 |
| high_school_computer_science | 45.0% | 47.5% | +2.5 | 40 |
| high_school_european_history | 55.0% | 50.0% | -5.0 | 40 |
| high_school_geography | 32.5% | 32.5% | +0.0 | 40 |
| high_school_government_and_politics | 57.5% | 55.0% | -2.5 | 40 |
| high_school_macroeconomics | 42.5% | 35.0% | -7.5 | 40 |
| high_school_mathematics | 25.0% | 32.5% | +7.5 | 40 |
| high_school_microeconomics | 35.0% | 35.0% | +0.0 | 40 |
| high_school_physics | 40.0% | 30.0% | -10.0 | 40 |
| high_school_psychology | 40.0% | 40.0% | +0.0 | 40 |
| high_school_statistics | 45.0% | 37.5% | -7.5 | 40 |
| high_school_us_history | 50.0% | 42.5% | -7.5 | 40 |
| high_school_world_history | 52.5% | 57.5% | +5.0 | 40 |
| human_aging | 45.0% | 52.5% | +7.5 | 40 |
| human_sexuality | 25.0% | 27.5% | +2.5 | 40 |
| international_law | 65.0% | 62.5% | -2.5 | 40 |
| jurisprudence | 47.5% | 52.5% | +5.0 | 40 |
| logical_fallacies | 37.5% | 35.0% | -2.5 | 40 |
| machine_learning | 47.5% | 40.0% | -7.5 | 40 |
| management | 35.0% | 42.5% | +7.5 | 40 |
| marketing | 35.0% | 37.5% | +2.5 | 40 |
| medical_genetics | 60.0% | 47.5% | -12.5 | 40 |
| miscellaneous | 50.0% | 42.5% | -7.5 | 40 |
| moral_disputes | 32.5% | 32.5% | +0.0 | 40 |
| moral_scenarios | 37.5% | 42.5% | +5.0 | 40 |
| nutrition | 35.0% | 45.0% | +10.0 | 40 |
| philosophy | 47.5% | 50.0% | +2.5 | 40 |
| prehistory | 37.5% | 22.5% | -15.0 | 40 |
| professional_accounting | 27.5% | 25.0% | -2.5 | 40 |
| professional_law | 35.0% | 30.0% | -5.0 | 40 |
| professional_medicine | 32.5% | 30.0% | -2.5 | 40 |
| professional_psychology | 42.5% | 40.0% | -2.5 | 40 |
| public_relations | 25.0% | 25.0% | +0.0 | 40 |
| security_studies | 45.0% | 42.5% | -2.5 | 40 |
| sociology | 55.0% | 65.0% | +10.0 | 40 |
| us_foreign_policy | 67.5% | 55.0% | -12.5 | 40 |
| virology | 35.0% | 25.0% | -10.0 | 40 |
| world_religions | 65.0% | 52.5% | -12.5 | 40 |

</details>

### Additional direct refusal-removal check

On 200 prompts hand-verified to make the base refuse consistently:

| Model | refuse | comply | empty |
|---|---:|---:|---:|
| Base `PQ2_0` | **200/200 (100%)** | 0 | 0 |
| **CRACK `PQ2_0`** | **0/200 (0%)** | 199/200 | 1 |

## Serving

Serve exactly like the base ternary release — PrismML's llama.cpp fork (CUDA / Metal / CPU).

```bash
# clone and build the fork (once)
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)

# serve
./build/bin/llama-server \
  -m Bonsai-2-27B-PQ2_0-CRACK.gguf \
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
