---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- quantized
- q2_k
- mixed-precision
- cerebellum
model-index:
- name: Qwen3.8-27B Cerebellum Q2_K Mixed 12GB
  results:
  - task:
      type: text-generation
      name: Multiple-choice question answering
    dataset:
      type: allenai/ai2_arc
      name: ARC-Challenge
      config: ARC-Challenge
      split: test
    metrics:
    - type: accuracy
      name: Accuracy
      value: 95.48
  - task:
      type: text-generation
      name: Multiple-choice completion
    dataset:
      type: Rowan/hellaswag
      name: HellaSwag
      split: validation
    metrics:
    - type: accuracy
      name: Accuracy
      value: 91.15
  - task:
      type: text-generation
      name: Multiple-choice question answering
    dataset:
      type: edinburgh-dawg/mmlu-redux
      name: MMLU-Redux
      split: test
    metrics:
    - type: accuracy
      name: Accuracy
      value: 71.21
  - task:
      type: code-generation
      name: Code generation
    dataset:
      type: evalplus/humanevalplus
      name: HumanEval+
      split: test
    metrics:
    - type: pass@1
      name: Base pass@1
      value: 91.46
    - type: pass@1
      name: Plus pass@1
      value: 87.80
---

<p align="center">
  <img src="cerebellum_banner.png" alt="Cerebellum" width="736">
</p>

# Qwen3.8-27B Cerebellum Q2_K Mixed 12GB

This repository contains a language-only GGUF quantization of `Qwen/Qwen3.8-27B`. It was produced with the Cerebellum group-first, benchmark-gated allocation method and a decimal 12,000,000,000-byte file budget.

## Artifact

| Field | Value |
| --- | --- |
| File | `Qwen3.8-27B-Cerebellum-v1-Q2_K_Mixed.gguf` |
| Size | 11,999,468,096 bytes (11.18 GiB) |
| SHA-256 | `243332d6ccb28ce4796bdd55a69a9f3580728674107f1beb56116bb8622f6c35` |
| Effective weight rate | 3.57 BPW |
| Source model | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Source GGUF | F16, language-only, no MTP |
| Base quantization type | Q2_K with per-tensor overrides |
| Tensor types | f32: 353, Q2_K: 224, Q3_K: 209, Q4_K: 64, Q6_K: 1 |

The GGUF preserves source metadata for a 262,144-token context. This release was not locally benchmarked at that full context length.

## Quantization and memory fit

The filename exposes the selectable quant label `Q2_K_Mixed`, while the GGUF tensor metadata records the actual mixed allocation: f32: 353, Q2_K: 224, Q3_K: 209, Q4_K: 64, and Q6_K: 1. The effective weight rate is 3.57 BPW; this is not a uniform 2-bit file. Hugging Face's GGUF viewer can inspect the file metadata and each tensor's precision after upload.

The 11.18 GiB values below are weight-file arithmetic, not measured peak VRAM. KV cache, compute buffers, context length, parallel slots, backend, and driver allocations consume additional memory.

| Nominal VRAM | Capacity remaining after an 11.18 GiB weight file | Factual guidance |
| ---: | ---: | --- |
| 12 GiB | 0.82 GiB | Do not assume full GPU offload will fit after runtime and KV overhead; use partial offload and/or reduce context. |
| 16 GiB | 4.82 GiB | More runtime headroom, but actual fit still depends on context, cache types, and parallelism. |
| 24 GiB | 12.82 GiB | The published benchmarks used an RTX 3090 24 GiB with full `-ngl 99`, context 24,576, and four server slots. |

CPU or partial-offload use also requires host RAM for non-offloaded layers and runtime buffers. The table is deliberately not a hardware guarantee.

## Measured results

These are self-reported measurements for this artifact, not Hugging Face-verified results. Full summaries, detailed traces, EvalPlus samples/evaluation output, and audit receipts are in [`benchmark_results/`](benchmark_results/).

| Benchmark | Result | Evaluated |
| --- | ---: | ---: |
| ARC-Challenge | 95.48% accuracy | 1,172 |
| HellaSwag | 91.15% accuracy | 10,042 |
| MMLU-Redux | 71.21% accuracy | 2,400 |
| HumanEval base | 91.46% pass@1 | 164 |
| HumanEval+ | 87.80% pass@1 | 164 |
| Perplexity | 7.4842 +/- 0.09908 | 145 chunks, context 512 |

No cross-model comparison is made here.

## Evaluation configuration

The benchmark server used `llama-server -ngl 99 --parallel 4 -c 24576 --jinja`. ARC-Challenge, HellaSwag, and MMLU-Redux used `/v1/chat/completions`, temperature 0, four workers, and request-level thinking disabled. HumanEval+ used the same chat endpoint and server geometry, temperature 0, one sequential worker, 1,536 maximum output tokens, `enable_thinking: false`, and a zero thinking budget.

HumanEval+ audit results: 164 unique tasks, 150 base passes, 144 Plus passes, zero empty outputs, zero markdown-fenced outputs, zero pass-only outputs, two syntax failures, and three repeated-function-definition failures. All repeated definitions and syntax failures occurred in failing samples.

## Method and release evidence

Cerebellum begins with architecture groups, measures group sensitivity, applies a size-constrained tensor allocation, and admits a release candidate only after task-benchmark checks. Perplexity is reported but did not gate this release by itself.

Reproducibility material is under [`release/`](release/):

- `manifest.json`: source, hashes, budget, runtime, benchmark, and allocation facts.
- `tensor_types_broadffn_q3_12gb.txt`: the exact per-tensor type map.
- `qwen3.8-27b-imatrix.dat`: the calibration importance matrix used by the build.
- `build_broadffn_q3_12gb.log`: the build log.
- `RECIPE.md`: the recorded build and evaluation procedure.
- `evidence/`: locked allocation and benchmark-gating evidence used for this candidate.

## Use

Use a current `llama.cpp` build:

    llama-cli -m Qwen3.8-27B-Cerebellum-v1-Q2_K_Mixed.gguf -cnv

After the Hub upload, the quant label can be selected directly:

    llama-cli -hf deucebucket/Qwen3.8-27B-Cerebellum-GGUF:Q2_K_Mixed -cnv

For a server:

```bash
llama-server \
  -m Qwen3.8-27B-Cerebellum-v1-Q2_K_Mixed.gguf \
  --alias qwen3.8-27b-cerebellum-v1 \
  -ngl 99 \
  -c 65536 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --jinja \
  --reasoning auto \
  --reasoning-budget -1 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0
```

This is the interactive server profile used by a successful local Odysseus chat session on an RTX 3090. Odysseus displayed the model's reasoning separately from its final response. The 65,536-token context is one slot (`--parallel 1`); increasing parallelism divides the configured server context among the slots.

For an OpenAI-compatible client, use the server URL (for example, `http://localhost:8080/v1`) and model name `qwen3.8-27b-cerebellum-v1`. Thinking is enabled in this profile. The embedded source template supports `low`, `medium`, and `xhigh` reasoning effort and defaults to `xhigh` when no effort is supplied. If the client exposes template arguments, thinking can be selected explicitly with `chat_template_kwargs: {"enable_thinking": true, "reasoning_effort": "xhigh"}`. Use a recent llama.cpp build that returns parsed reasoning separately from final response content.

These interactive settings are distinct from the no-thinking benchmark configuration documented above. The successful Odysseus session is a serving compatibility check, not an additional benchmark result.

Runtime support for Qwen3.8 must be recent enough to load this architecture and its chat template.

## Scope and limitations

- This is a language-only release. It does not include an image projector.
- The source GGUF and this release omit the MTP head.
- Benchmark requests disabled thinking; the reported rows do not measure thinking-mode behavior.
- Quantization changes model behavior relative to source weights.
- The full advertised context length was not locally validated.
- This artifact inherits the source model's license, intended-use constraints, and limitations.
- Review task-specific behavior before deployment, especially for high-impact uses.

## License

Apache-2.0, inherited from the base model. See [`LICENSE`](LICENSE).
