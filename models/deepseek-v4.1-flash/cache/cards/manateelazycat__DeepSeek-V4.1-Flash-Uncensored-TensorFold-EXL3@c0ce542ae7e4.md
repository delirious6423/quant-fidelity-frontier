---
license: mit
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
- deepseek
- deepseek-v4.1
- abliterated
- uncensored
- crack
- multimodal
- moe
- exl3
- dgx-spark
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
thumbnail: dealign_mascot.png
---

<div align="center">

<img src="dealign_mascot.png" width="128" />

# DeepSeek-V4.1-Flash — UNCENSORED · EXL3 2.9bpw

**Abliterated · No guardrails · EXL3 2.9 bpw · Runs on 2× DGX Spark · Vision + tools + DSpark**

[@dealignai](https://x.com/dealignai)

</div>

---

## What is this

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) quantized to **EXL3 2.9 bits/weight** and **abliterated** — the safety guardrails are surgically removed at the weight level while capability, vision, reasoning, speculative decoding (DSpark) and multi-turn coherence are preserved. The 552B multimodal MoE now fits and serves on **two NVIDIA DGX Spark (GB10)** boxes.

**Proprietary weight-level abliteration** developed by the dealignai research team. No custom `model.py`, no runtime hooks, no steering vectors — a standard EXL3 checkpoint that loads exactly like the base quant. The refusal circuitry is removed while every capability-critical component (routed experts, Engram n-gram memory, CSA2 sparse attention, DSpark draft head, vision tower, router gates, norms, embeddings) is preserved.

| | |
|---|---|
| **Base** | `deepseek-ai/DeepSeek-V4.1-Flash` (552B backbone, 8B/16B active per token) |
| **Architecture** | Causal Encoder-Decoder (20+20), MoE (384 routed top-6 + 1 shared), Hyper-Connections, CSA2 sparse attention, Engram memory, DSpark speculative draft, vision tower |
| **Quant** | EXL3 trellis, mul1 codebook, average **2.9 bpw**, head 6-bit, MTP 4-bit |
| **Footprint** | ~197 GiB — fits 2× DGX Spark (GB10, 128 GiB each) at TP=2 |
| **Context** | up to 1M tokens (validated at 256k–600k on 2× Spark) |
| **Vision** | DeepSeek-ViT — preserved |
| **Speculation** | DSpark in-checkpoint draft — **works** (~45% acceptance, verified) |

## Results

Refusal graded on the **delivered output tokens** (content, or the reasoning
trace when the model reasons past the token budget) across a 6-tier scheme
(hard refusal / soft redirect / hedge / truncated-comply / reasoning-refusal /
comply). Truncation is never miscounted as a refusal.

### HarmBench-320 — attack success rate (comply %), T=0 greedy

| eval | base ASR | **CRACK ASR** |
|---|---:|---:|
| **HB-320 effort=off** | 36.1 % | **99.4 %** |
| **HB-320 effort=max** | 21.0 % | **99.4 %** |

_Base HB measured on a 287-item representative sample (off n=144, max n=143); CRACK on the full 320 each. MMLU is the full 14,042-item set for both._

At effort=max the **base** refuses even harder (reasoning surfaces safety
concerns first); the cracked build stays at 99.4 % across both effort levels.

**Per-category ASR (comply %) — all 7 HarmBench semantic categories:**

| category | base off | CRACK off | base max | CRACK max |
|---|---:|---:|---:|---:|
| chemical_biological | 7 % | **100 %** | 0 % | **100 %** |
| copyright | 95 % | **100 %** | 59 % | **99 %** |
| cybercrime_intrusion | 21 % | **100 %** | 0 % | **100 %** |
| harassment_bullying | 20 % | **100 %** | 0 % | **95 %** |
| harmful | 12 % | **94 %** | 0 % | **100 %** |
| illegal | 0 % | **98 %** | 7 % | **100 %** |
| misinformation_disinformation | 36 % | **100 %** | 27 % | **100 %** |

### MMLU-14k (full test set, base-logit ranking, T=0)

| build | acc | Δ |
|---|---:|---:|
| base (EXL3 2.9bpw) | **82.15 %** | — |
| **CRACK** | **79.20 %** | **−2.95 pp** |

The capability cost is concentrated almost entirely in the ethics cluster —
the same "should I refuse?" circuit that is removed:

| subset | base | CRACK | Δ |
|---|---:|---:|---:|
| **non-ethics** (n≈11,059) | 84.97 % | **84.39 %** | **−0.58 pp** |
| ethics cluster (n=2,983) | 71.67 % | 59.94 % | −11.73 pp |

General capability is essentially intact (−0.58 pp). The single largest
per-subject move is `moral_scenarios` (66.1 % → 37.8 %) — the refusal circuit
itself. Several subjects are unchanged or improved.

<details>
<summary><b>Full MMLU per-subject comparison (all 57 subjects, base → CRACK)</b></summary>

| subject | base | CRACK | Δpp | n |
|---|---:|---:|---:|---:|
| abstract_algebra | 70.0% | 74.0% | +4.0 | 100 |
| anatomy | 78.5% | 76.3% | -2.2 | 135 |
| astronomy | 93.4% | 92.8% | -0.7 | 152 |
| business_ethics | 80.0% | 81.0% | +1.0 | 100 |
| clinical_knowledge | 88.7% | 85.7% | -3.0 | 265 |
| college_biology | 94.4% | 91.0% | -3.5 | 144 |
| college_chemistry | 68.0% | 66.0% | -2.0 | 100 |
| college_computer_science | 79.0% | 76.0% | -3.0 | 100 |
| college_mathematics | 69.0% | 69.0% | +0.0 | 100 |
| college_medicine | 77.5% | 76.9% | -0.6 | 173 |
| college_physics | 86.3% | 87.3% | +1.0 | 102 |
| computer_security | 83.0% | 84.0% | +1.0 | 100 |
| conceptual_physics | 87.2% | 86.8% | -0.4 | 235 |
| econometrics | 73.7% | 73.7% | +0.0 | 114 |
| electrical_engineering | 71.7% | 75.9% | +4.1 | 145 |
| elementary_mathematics | 92.3% | 92.1% | -0.3 | 378 |
| formal_logic | 65.9% | 65.1% | -0.8 | 126 |
| global_facts | 64.0% | 59.0% | -5.0 | 100 |
| high_school_biology | 93.2% | 92.3% | -1.0 | 310 |
| high_school_chemistry | 79.8% | 80.8% | +1.0 | 203 |
| high_school_computer_science | 96.0% | 95.0% | -1.0 | 100 |
| high_school_european_history | 86.7% | 85.5% | -1.2 | 165 |
| high_school_geography | 89.9% | 89.9% | +0.0 | 198 |
| high_school_government_and_politics | 92.7% | 93.3% | +0.5 | 193 |
| high_school_macroeconomics | 88.2% | 88.2% | +0.0 | 390 |
| high_school_mathematics | 67.0% | 67.4% | +0.4 | 270 |
| high_school_microeconomics | 93.3% | 92.0% | -1.3 | 238 |
| high_school_physics | 78.1% | 82.1% | +4.0 | 151 |
| high_school_psychology | 92.1% | 93.4% | +1.3 | 545 |
| high_school_statistics | 85.2% | 81.9% | -3.2 | 216 |
| high_school_us_history | 92.2% | 91.7% | -0.5 | 204 |
| high_school_world_history | 91.6% | 92.0% | +0.4 | 237 |
| human_aging | 78.5% | 77.1% | -1.3 | 223 |
| human_sexuality | 82.4% | 83.2% | +0.8 | 131 |
| international_law | 85.1% | 88.4% | +3.3 | 121 |
| jurisprudence | 87.0% | 87.0% | +0.0 | 108 |
| logical_fallacies | 89.6% | 89.0% | -0.6 | 163 |
| machine_learning | 64.3% | 65.2% | +0.9 | 112 |
| management | 88.3% | 88.3% | +0.0 | 103 |
| marketing | 93.2% | 92.7% | -0.4 | 234 |
| medical_genetics | 96.0% | 92.0% | -4.0 | 100 |
| miscellaneous | 93.9% | 94.3% | +0.4 | 783 |
| moral_disputes | 80.1% | 76.0% | -4.0 | 346 |
| moral_scenarios | 66.1% | 37.8% | -28.4 | 895 |
| nutrition | 83.0% | 83.7% | +0.7 | 306 |
| philosophy | 84.6% | 83.9% | -0.6 | 311 |
| prehistory | 87.7% | 86.1% | -1.5 | 324 |
| professional_accounting | 72.3% | 67.7% | -4.6 | 282 |
| professional_law | 71.4% | 66.0% | -5.4 | 1534 |
| professional_medicine | 91.5% | 90.4% | -1.1 | 272 |
| professional_psychology | 83.7% | 80.7% | -2.9 | 612 |
| public_relations | 69.1% | 70.9% | +1.8 | 110 |
| security_studies | 83.3% | 80.8% | -2.4 | 245 |
| sociology | 89.1% | 86.1% | -3.0 | 201 |
| us_foreign_policy | 92.0% | 91.0% | -1.0 | 100 |
| virology | 51.8% | 53.6% | +1.8 | 166 |
| world_religions | 86.5% | 87.1% | +0.6 | 171 |

</details>

## Serving (2× DGX Spark)

**Runtime:** [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
— a vLLM + ExLlamaV3 (EXL3) overlay image (`ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw`)
that carries the `DeepseekV41` architecture and the SM121 (GB10) kernels.

This checkpoint is a drop-in replacement for the kit's stock 2.9bpw pack — point
`MODEL_HOST` at it and serve; nothing else changes.

**Two things you must supply:**

1. **Engram tables** — shards **47 + 48** of `deepseek-ai/DeepSeek-V4.1-Flash`
   (the n-gram tables are never quantized and are not duplicated here). Point
   the kit's `ENGRAM_DIR` at a tree containing those two shards + the index.
2. A **2× GB10** kit joined over CX7 (the pack is ~197 GiB, TP=2).

**Quick start** (on the head node, from the kit repo):

```bash
cp .env.example .env
# edit .env:
MODEL_HOST=/path/to/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw
ENGRAM_DIR=/path/to/engram-src        # shards 47+48 of the base model
AUTO_DOWNLOAD=0
SKIP_BUILD=1 ./start.sh                # pull the published image + serve on :8888
```

**Serving params that work** (validated on 2× GB10, TP=2):

| flag | value | note |
|---|---|---|
| `QUANTIZATION` | `exl3` | EXL3 trellis, mul1 codebook |
| `TP` / `NNODES` | `2` / `2` | tensor-parallel over CX7 |
| `SPEC_METHOD` | `dspark` | in-checkpoint speculative draft — **works** |
| `DSPARK_TOKENS` | `3` | k=3 (measured faster than k=5 on prose) |
| `MAX_MODEL_LEN` | `262144` | 256k tested here; the kit validates up to 600k |
| `KV_CACHE_MEMORY_BYTES` | `1073741824` | 1 GiB pinned KV pool |
| `KV_BLOCK_SIZE` | `64` | SM12x indexer takes 32/64 (not 128) |
| `GPU_MEM_UTIL` | `0.85` | cap ≤ 0.85 on GB10 |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | ≥ 1536 required with the vision tower on |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | `256` | must be set (empty → `int('')` crash after load) |
| `LANGUAGE_MODEL_ONLY` | `0` | vision on (set `1` for text-only) |

**Sampling** — official DeepSeek settings: `temperature=1.0`, `top_p=0.95`.
Thinking defaults on; set `reasoning_effort` to `"low"` / `"high"` / `"max"`
(or `enable_thinking=false` for no reasoning). API is OpenAI-compatible on
`:8888`, served model id `DeepSeek-v4.1-Flash-EXL3`.

**Speculative decoding (DSpark) works on this abliterated build** — measured
~45 % draft acceptance (mean accept length ~2), the same on harmful and
harmless prompts, ~27 tok/s single-stream on 2× GB10. Vision, tools and long
context are unchanged from the base pack.

## Preserved (byte-compatible with the base quant)

Routed experts · Engram memory · CSA2 sparse attention · DSpark draft head ·
vision tower · router gates · RMSNorms · embeddings. Only the refusal circuit
is removed.

## Responsible use

This model has its safety guardrails removed. You are responsible for your
inputs and outputs and for complying with all applicable law.
