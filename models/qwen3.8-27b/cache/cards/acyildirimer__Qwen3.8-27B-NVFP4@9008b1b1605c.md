---
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - qwen3.8
  - vllm
  - modelopt
  - nvfp4
  - fp8
  - quantized
  - conversational
---

# Qwen3.8-27B-NVFP4

Mixed-precision NVIDIA ModelOpt quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), optimized for
NVIDIA Blackwell GPUs. It is based on upstream revision
[`1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).

The goal of this quantization is to substantially reduce the memory footprint of Qwen3.8-27B while retaining model quality by keeping quantization-sensitive components at higher precision.

## Quantization

This is a mixed NVFP4 / FP8 / BF16 checkpoint rather than a uniformly 4-bit model.

| Component | Precision |
|---|---|
| MLP layers 0–55 | W4A16 NVFP4, group size 16 |
| MLP layers 56–63 | FP8 W8A8 |
| Full-attention projections | FP8 W8A8 |
| Large linear-attention projections | FP8 W8A8 |
| `lm_head` | FP8 W8A8 |
| KV cache | FP8 scaling metadata; intended for an FP8 runtime cache |
| Vision tower | BF16 |
| MTP head | BF16 |

Quantization was performed with **NVIDIA ModelOpt 0.45.0**.

No fine-tuning or additional training was performed.

## Calibration

Calibration used **1,024 fully populated 512-token samples** from a
deterministic version-5 corpus (seed `42`). The source mix was:

| Calibration source | Share |
|---|---:|
| [Nemotron SFT Instruction Following Chat v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | 17% |
| [Nemotron Science v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | 8% |
| [Nemotron Competitive Programming v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | 10% |
| [Nemotron SFT Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) — tool calling | 7% |
| [Nemotron SFT Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) — web search | 3% |
| [Nemotron Math v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | 13% |
| [Nemotron SFT SWE v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) | 10% |
| [Nemotron SFT Multilingual v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v1) | 12% |
| [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) | 20% |

The mixed-precision layout was selected to keep more quantization-sensitive parts of the Qwen3.8 hybrid architecture at FP8 or BF16 while applying NVFP4 to the majority of the MLP weights.

## Quality Benchmarks

Quality was evaluated with `llm-bench-rig` through the OpenAI-compatible vLLM API.

The comparison below uses **specific released Qwen3.8-27B checkpoints**, rather than generic quantization labels.

Thinking was disabled during these quality benchmark runs to keep the evaluation
conditions consistent and avoid mixing reasoning-mode differences into the
checkpoint comparison.

| Checkpoint | Quantization | Backend | MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | 4-task mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **acyildirimer/Qwen3.8-27B-NVFP4** | **ModelOpt NVFP4 + FP8** | **vLLM** | **85.1** | **96.8** | **94.1** | **96.9** | **92.1** | **93.2** |
| [acyildirimer/Qwen3.8-27B-AutoQuant-NVFP4-FP8-5bit](https://huggingface.co/acyildirimer/Qwen3.8-27B-AutoQuant-NVFP4-FP8-5bit) | ModelOpt NVFP4 + FP8, 5-bit target | vLLM | — | **96.25** | — | **96.82** | **93.29** | — |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | BF16 | llama.cpp (partial offload) | 85.3 | 96.8 | 94.3 | 97.4 | 93.9 | 93.5 |
| [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | Dynamic V3 NVFP4 | vLLM | 84.3 | 96.9 | 94.3 | 97.1 | 89.6 | 93.2 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q8_0 | llama.cpp | 85.2 | 96.8 | 94.4 | 97.4 | 94.5 | 93.5 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q6_K | llama.cpp | 85.3 | 96.7 | 94.3 | 97.5 | 94.5 | 93.5 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL | llama.cpp | 85.1 | 96.6 | 94.4 | 97.3 | 93.9 | 93.4 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q4_K_M | llama.cpp | 85.0 | 96.8 | 94.3 | 97.1 | 92.7 | 93.3 |

`4-task mean` is the arithmetic mean of MMLU, ARC-Challenge, HellaSwag, and GSM8K. HumanEval is reported separately as executable pass@1 and is not included in that aggregate.

The 5-bit checkpoint was evaluated on ARC-Challenge, GSM8K, and HumanEval. Its
arithmetic mean across those three reported scores is **95.45**. MMLU and
HellaSwag results were not reported for that checkpoint, so a four-task mean is
not shown.

The 5-bit results are author-reported. The remaining comparison results come
from the public
[`witcheer/rtx-5090-benchmarks`](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks)
dataset and use the same `llm-bench-rig` evaluator. The BF16 and GGUF
checkpoints were served with llama.cpp, while the NVFP4 checkpoints were served
through vLLM. The protocol is common, but the serving stacks differ, so these
figures are indicative comparisons rather than a controlled quantization
ablation.

The official [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
checkpoint is not shown because a result from this exact protocol is not
currently available. Scores from other evaluation protocols are not mixed into
the table.

### Evaluation protocol

| Benchmark | Configuration |
|---|---|
| MMLU | 5-shot, 50% stratified sample |
| ARC-Challenge | 25-shot, full 1,172 examples |
| HellaSwag | 10-shot, 50% stratified sample |
| GSM8K | 5-shot CoT, full 1,319 examples |
| HumanEval | 0-shot, full 164 problems where available |

Common settings:

- Thinking: disabled for all quality benchmarks in this section
- Temperature: `0`
- MMLU and HellaSwag sampling seed: `42`
- Multiple-choice tasks use generative answer extraction rather than log-likelihood scoring
- GSM8K uses exact numeric answer matching
- HumanEval uses executable pass@1 evaluation

Because the multiple-choice evaluations are generative, the absolute scores should not be compared directly with results from `lm-evaluation-harness` log-likelihood evaluation.

### Results

Across the four multiple-choice and math benchmarks, this checkpoint reaches a **93.2% mean**, compared with:

- **93.5%** for the BF16 reference
- **93.5%** for Unsloth Q8_0
- **93.5%** for Unsloth Q6_K
- **93.4%** for Unsloth UD-Q4_K_XL
- **93.3%** for Unsloth Q4_K_M
- **93.2%** for Unsloth Dynamic V3 NVFP4

The largest absolute difference from the BF16 reference across those four tests
is **0.5 percentage points** on GSM8K. MMLU, ARC-Challenge, and HellaSwag are
within **0.2 percentage points**.

On HumanEval, this checkpoint achieves **92.1% pass@1 (151/164)**. This is 1.8
percentage points below the BF16 reference and 2.5 points above the listed
Unsloth Dynamic V3 NVFP4 result under the same evaluation protocol.

The 5-bit checkpoint achieves **93.29% HumanEval pass@1** under the reported
evaluation.

## Tool-Calling Evaluation

The checkpoint was also evaluated with `tool-eval-bench` **v2.2.0** using:

- vLLM 0.27.1
- temperature `0`
- seed `42`
- all 69 scenarios

| Metric | Result |
|---|---:|
| Checks passed | **125 / 138** |
| Normalized score | **91 / 100** |
| Failed prompt-injection scenarios | **TC-34, TC-58** |

Tool-enabled deployments should therefore apply independent authorization, permission checks, and defenses for untrusted tool or web content rather than relying on model behavior alone.

## vLLM Usage

A runtime with support for mixed NVIDIA ModelOpt **W4A16 NVFP4 + FP8**
checkpoints is required. The examples below use the vLLM 0.27.1 command-line
interface.

### Text only

```bash
vllm serve acyildirimer/Qwen3.8-27B-NVFP4 \
    --quantization modelopt \
    --max-model-len auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --performance-mode interactivity \
    --language-model-only \
    --attention-backend auto \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --generation-config auto \
    --override-generation-config \
      '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --tool-call-parser qwen3_coder
```

### Text only with MTP

```bash
vllm serve acyildirimer/Qwen3.8-27B-NVFP4 \
    --quantization modelopt \
    --max-model-len auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --performance-mode interactivity \
    --language-model-only \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --attention-backend auto \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --generation-config auto \
    --override-generation-config \
      '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --tool-call-parser qwen3_coder
```

### Text, vision, and MTP

```bash
vllm serve acyildirimer/Qwen3.8-27B-NVFP4 \
    --quantization modelopt \
    --max-model-len auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.91 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --performance-mode interactivity \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --attention-backend auto \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --generation-config auto \
    --override-generation-config \
      '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --tool-call-parser qwen3_coder
```

Memory-utilization values are examples and should be adjusted for the available GPU memory, context length, concurrency, vision use, and speculative-decoding configuration.

MTP does not improve every hardware and workload combination. Compare latency
and throughput with MTP enabled and disabled, and inspect vLLM's speculative
decoding metrics (including the average draft acceptance rate) before choosing
a production configuration.

## Qwen3.8 Features

This checkpoint preserves the architecture and capabilities of the upstream Qwen3.8-27B model, including:

- text generation
- image and video understanding
- configurable thinking behavior
- tool calling
- long-context inference
- Multi-Token Prediction (MTP)

The upstream model has a native context length of **262,144 tokens** and supports extension beyond the native context using the mechanisms documented by Qwen.

Refer to the upstream
[Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the
authoritative description of architecture, capabilities, recommended generation
settings, context extension, vision preprocessing, and limitations.

## Hardware

NVFP4 is intended primarily for NVIDIA hardware with native FP4 support,
particularly **Blackwell-generation GPUs**. vLLM can fall back to weight-only
W4A16 execution on some GPUs without a native FP4 GEMM kernel, but this can
reduce throughput. The runtime must also support the checkpoint's FP8 layers.

Actual performance depends heavily on:

- GPU architecture
- runtime and kernel implementation
- tensor-parallel topology
- batch size and concurrency
- context length
- KV-cache configuration
- MTP/speculative decoding
- attention backend

This repository provides model weights rather than universal performance guarantees for a particular hardware configuration.

## Limitations

This is a lossy post-training quantization of the original BF16 checkpoint.

The benchmark results above show strong preservation on the evaluated short-context tasks, but they do not guarantee identical behavior to BF16 across:

- long-context workloads
- difficult coding and agentic tasks
- multimodal inputs
- uncommon languages
- very long generations
- tool-use workflows
- safety-sensitive applications

In particular, short benchmark suites can understate quantization differences that emerge under long-context or complex agentic workloads.

The model may also inherit inaccuracies, biases, undesirable outputs, and other limitations from the upstream model.

## Acknowledgements

This checkpoint is derived from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and retains the
upstream **Apache 2.0** license.

Quantization was performed with
[NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer).

Comparison results for the Unsloth NVFP4 and GGUF checkpoints are based on public `llm-bench-rig` measurements from the `witcheer/rtx-5090-benchmarks` project.

See the upstream Qwen model card for model authorship, training information,
intended use, and citation information.
