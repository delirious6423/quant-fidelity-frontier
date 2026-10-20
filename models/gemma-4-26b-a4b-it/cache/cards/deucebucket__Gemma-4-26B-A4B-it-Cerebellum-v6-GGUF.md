---
license: gemma
library_name: gguf
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
model_name: Gemma-4-26B-A4B-it-Cerebellum-v6.1-templatefix-GGUF
model_creator: google
model_type: gemma4
quantized_by: deucebucket
pipeline_tag: text-generation
tags:
  - GGUF
  - gemma4
  - gemma
  - google
  - quantized
  - cerebellum
  - imatrix
  - moe
  - 3-bit
  - templatefix
model-index:
- name: Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF
  results:
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: AI2 Reasoning Challenge
      type: ai2_arc
      config: ARC-Challenge
      split: test
    metrics:
    - name: normalized accuracy
      type: acc_norm
      value: 0.9556
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HellaSwag
      type: hellaswag
      split: validation
    metrics:
    - name: accuracy
      type: acc
      value: 0.8455
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: MMLU-Redux
      type: cais/mmlu
      config: all
      split: test
    metrics:
    - name: accuracy
      type: acc
      value: 0.7133
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF/tree/main/benchmark_results
---

<p align="center">
  <img src="cerebellum_banner.png" alt="Cerebellum" width="640">
</p>

# Gemma 4 26B-A4B-it Cerebellum GGUF

Sensitivity-guided mixed-precision GGUF of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it):
a Q3_K_M base with the Cerebellum v6 tensor allocation. The shipped file carries
the v6 weights plus Google's updated Gemma 4 chat-template metadata (the 2026-05-18
template state) with zero tensor changes versus v6. Newer versions appear in
filenames, not the repo name.

## Files

| File | Description |
|------|-------------|
| `gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix-Q3_K_M.gguf` | ~11 GB; v6 allocation + updated chat-template metadata |
| `gemma-4-26b-a4b-it.mmproj.gguf` | vision projector (required for image/video) |

## Evaluation

Measured directly on the GGUF with llama.cpp `llama-server` on an RTX 3090,
temperature 0, project benchmark harness. v6.1 is metadata-only over v6, so these
describe the same weights. The comparison column is our own same-size uniform
Q3_K_M build measured on the same harness. Summary JSONs are in `benchmark_results/`.

| Benchmark | Cerebellum v6 (11 GB) | Uniform Q3_K_M (11 GB) |
|-----------|:---:|:---:|
| ARC-Challenge (1172 q) | 95.56% | 95.22% |
| HellaSwag (10042 q) | 84.55% | 86.57% |
| MMLU-Redux (2400 q) | 71.33% | 73.67% |
| HumanEval (raw-completions, legacy) | pending re-audit | 62.2% pass@1 |

HumanEval for Gemma 4 must use the chat-completions harness
(`scripts/benchmark_evalplus_chat.py`, `enable_thinking: false`,
`thinking_budget_tokens: 0`, `BENCH_WORKERS=1`). The retained v6 HumanEval
artifacts were raw-completions and are marked for re-audit, so no v6 HumanEval
number is published here.

## Usage

Gemma 4 requires `--jinja`. For non-thinking output, pass request-level
`chat_template_kwargs: {"enable_thinking": false}` and `thinking_budget_tokens: 0`;
do not set a fixed server `--reasoning-budget` (it can burn output into hidden
reasoning until the length cap, which looks like a repetition loop).

```bash
llama-server \
  --model gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix-Q3_K_M.gguf \
  --mmproj gemma-4-26b-a4b-it.mmproj.gguf \
  -ngl 99 --ctx-size 65536 --parallel 1 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --reasoning auto
```

Measured on one RTX 3090 (24 GB), KV q8_0: ~123 tok/s decode, 15.1 GB peak VRAM
(4-slot serving), context to 131,072. This rig's measurements; no quality claims
beyond them.

## Provenance

- Base: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) — Google Gemma Team
- Base quant lineage: Q3_K_M with the bartowski imatrix (`bartowski/google_gemma-4-26B-A4B-it-GGUF`)
- Recipe: Cerebellum v6 tensor allocation; v6.1 is a chat-template metadata refresh
  (Google 2026-05-18 template), zero tensor changes

## Credits

- Base model: Google Gemma Team, `google/gemma-4-26B-A4B-it`
- Imatrix: bartowski, `bartowski/google_gemma-4-26B-A4B-it-GGUF`
- GGUF runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Quantization method: [Cerebellum](https://github.com/deucebucket/cerebellum) — deucebucket
