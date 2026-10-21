---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- moe
- pruning
- reap
- gguf
- quantized
- coding
language:
- en
pipeline_tag: text-generation
library_name: llama.cpp
model-index:
- name: Gemma4-26B-A4B-REAP-25-GGUF
  results:
  - task:
      type: text-generation
      name: Code Generation
    dataset:
      name: HumanEval
      type: openai_humaneval
    metrics:
    - type: pass@1
      value: 0.927
      name: pass@1 (Q3_K mixed)
  - task:
      type: text-generation
      name: Code Generation
    dataset:
      name: MBPP
      type: mbpp
    metrics:
    - type: pass@1
      value: 0.704
      name: pass@1 (Q3_K mixed)
  - task:
      type: text-generation
      name: Instruction Following
    dataset:
      name: IFEval
      type: HuggingFaceH4/ifeval
    metrics:
    - type: accuracy
      value: 0.804
      name: prompt-level (Q3_K mixed)
    - type: accuracy
      value: 0.866
      name: instruction-level (Q3_K mixed)
---

# Gemma 4 26B-A4B · REAP-25 · Q3_K / Q2_K mixed (imatrix) — GGUF

A **9–10 GB** mixture-of-experts specialist derived from
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) by
pruning 25 % of routed experts with routing-aware REAP and quantizing the
result with attention-aware mixed-precision schemes calibrated by
`llama-imatrix`.

Two quantizations are provided in this repo:

| File | Size | HumanEval | MBPP | When to pick |
|------|-----:|----------:|-----:|--------------|
| **`Gemma4-26B-A4B-REAP-25-Q3_K-mixed.gguf`** | 10.3 GB | **92.7 %** | 70.4 % | **Default.** Best quality/size trade |
| `Gemma4-26B-A4B-REAP-25-Q2_K-mixed.gguf` | 9.1 GB | 90.9 % | — | Tightest memory (headroom for KV / other apps) |

The Q4_K_M (13 GB) baseline scored 91.5 % HumanEval — bigger AND lower quality
than the Q3_K mixed variant, so it isn't shipped here.

Companion release to
[`crucible-labs/Qwen3.6-35B-A3B-REAP-48-Q3K-mixed-GGUF`](https://huggingface.co/crucible-labs/Qwen3.6-35B-A3B-REAP-48-Q3K-mixed-GGUF).

## TL;DR

| | Base · bf16 | REAP-25 · bf16 | **REAP-25 · Q3_K-mixed (this repo)** |
|---|---:|---:|---:|
| Parameters | ~26 B | ~19.5 B | ~19.5 B |
| File size on disk | ~52 GB | ~40 GB | **10.3 GB** |
| HumanEval | 93.9 % | 93.9 % | **92.7 %** |
| MBPP | 73.0 % | — | 70.4 % |
| IFEval (prompt / instr) | — | — | **80.4 % / 86.6 %** |
| MMLU-Pro (5-shot) | 82.6 %* | — | 76.6 % |

\* MMLU-Pro base score is from the Gemma 4 published model card, not
re-measured here. All other numbers are from this project's own harness.

## Method

### Compression: routing-aware REAP, 25 % expert pruning
Routed experts are scored by router-weighted activation magnitude *and* a
routing-disruption penalty (the routing-aware variant of REAP); the lowest-
scoring 25 % per layer are pruned. See arXiv:2510.13999 for the method.

**Why routing-aware here** (and not on Qwen 3.6): Gemma 4 has no shared expert
to absorb load when a routed expert is pruned, so the routing-disruption
penalty measures real quality cost rather than being double-counted by the
architecture. On models with a shared expert (Qwen 3.6, DeepSeek, etc.), plain
REAP without the routing term is sufficient.

Result before quantization: ~26 B → ~19.5 B parameters; bf16 size ~40 GB.

### Quantization: mixed profiles + imatrix
Both files use llama.cpp's `Q3_K_S` selection base with attention overridden
to Q8_0, calibrated with `llama-imatrix` on the same code / tool-use / math
mix used for REAP scoring. Approximate distribution (658 tensors total):

| Tensor group | Q3_K-mixed file | Q2_K-mixed file |
|---|---|---|
| Attention (attn_q/k/v/output), embeddings, output | Q8_0 (129 tensors) | Q8_0 (129 tensors) |
| Mid-precision body | Q4_K + Q5_0 + Q4_0 (106) | Q4_K + Q5_0 + Q4_0 (106) |
| Lowest-precision body | **Q3_K (30)** | **Q2_K (30)** |
| Layer norms / small | F32 (392) | F32 (392) |

The two files differ only in those 30 lowest-precision tensors — Q3_K vs Q2_K
— which is where the ~1 GB size difference comes from.

## Inference

### llama.cpp / llama-server (recommended)

```bash
llama-server \
  -m Gemma4-26B-A4B-REAP-25-Q3_K-mixed.gguf \
  --port 8093 --host 0.0.0.0 \
  -c 8192 -ngl 999 \
  --jinja --reasoning off
```

### Recommended sampling
- `temperature: 0` (deterministic) or `0.2` for variation
- `max_tokens: 2048` for code, higher for long-form

### Performance
Not re-measured here for the two shipping quants specifically. Reference
numbers from the same-family Q2_K mixed on M4 Mac Mini via Metal:
**32.6 tg / 401 pp**. Expect the Q3_K variant to be similar or slightly slower
in token generation, similar in prompt processing.

## Known issues

### Q2_K prompt processing is 7× slower on ROCm
llama.cpp's Q2_K dequant kernel on ROCm (gfx1151) has a known ~7× slowdown
in prompt processing versus other quant types. Metal (Mac) and CUDA (NVIDIA)
are not affected. If you're on an AMD GPU and PP throughput matters, pick the
Q3_K-mixed file instead.

### Gemma 4 compression cliff is at ~37 %
We measured earlier ratios of REAP on this model. Above ~30 % pruning, HumanEval
starts degrading noticeably; at 37.5 % pruning it dropped to 75.0 % — a hard
cliff. This is why the release ships at 25 % pruning even though Qwen 3.6
tolerates 48 %.

## Reproducibility

This artifact was produced by the Crucible compression toolkit (developed
alongside this model; source not yet publicly released — planned). High-level
pipeline (identical shape to the Qwen 3.6 release):

```bash
# 1. REAP scoring (routing-aware)
crucible observe google/gemma-4-26B-A4B-it \
  --samples 512 --seed 42 \
  -o results/gemma4-base-observation.json

# 2. REAP pruning at 25 %, routing-aware
crucible compress google/gemma-4-26B-A4B-it \
  --method reap --routing-aware --ratio 0.25 \
  --scores-file results/gemma4-base-observation.json \
  -o outputs/gemma4-reap-25pct

# 3-5. GGUF convert, imatrix, quantize (same steps as Qwen release)
```

## Evaluation methodology

Numbers above are from Crucible's own-harness `bench` subcommand (no
`evalplus` / `lm-eval` runtime dependency). Each problem is run once at
`temperature=0`; code is executed in a subprocess with a 30 s timeout.
HumanEval numbers use the standard 164 problems (vanilla, not the augmented
`humanevalplus`). MBPP numbers use the vanilla 500-problem split.
IFEval uses the standard HuggingFaceH4/ifeval dataset with fixed checkers
(paragraph counter that skips markdown dividers, sentence splitter on
punctuation + uppercase, whitespace-normalized repeat_prompt, etc.).

## Intended use & limitations

Designed as a coding specialist for **local single-user inference** on
consumer hardware.

Not designed for or measured on: multilingual generation, image/audio input
(the compressor operates only on the language-model component), long-form
prose, or code in languages other than Python.

The base Gemma 4 model has substantially different training and behavior
from Qwen 3.6 — pick the model that fits your use case:
- [Qwen 3.6 REAP-48 Q3K-mixed](https://huggingface.co/crucible-labs/Qwen3.6-35B-A3B-REAP-48-Q3K-mixed-GGUF):
  9.4 GB, stronger BFCL tool-calling (91.8 %), also has BigCodeBench
  numbers, 3 B active params
- **Gemma 4 REAP-25 Q3K-mixed (this repo)**: 10.3 GB, higher raw HumanEval
  (92.7 %), 4 B active params, Gemma-family instruction-following
  characteristics

## License

Apache-2.0, inherited from
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).

## Citation

```bibtex
@article{lasby2025reap,
  title={REAP the Experts: Why Pruning Prevails for One-Shot MoE compression},
  author={Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and
          Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
  journal={arXiv preprint arXiv:2510.13999},
  year={2025}
}
```
