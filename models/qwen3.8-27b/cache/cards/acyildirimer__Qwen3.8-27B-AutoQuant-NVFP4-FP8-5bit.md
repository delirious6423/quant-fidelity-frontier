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
  - autoquant
  - modelopt
  - nvfp4
  - fp8
  - mixed-precision
  - vllm
  - quantized
  - conversational
---

# Qwen3.8-27B NVFP4/FP8 5-bit

Mixed-precision NVIDIA ModelOpt quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), based on upstream
revision
[`1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).

This checkpoint targets a strong size-to-quality balance by assigning compatible
language projections to NVFP4 or FP8 according to measured sensitivity. No
fine-tuning or additional training was performed.

## Quantization

| Component | Precision |
|---|---|
| Language projections | 309 W4A16 NVFP4 modules + 92 FP8 W8A8 modules |
| `lm_head` | W4A16 NVFP4, group size 16 |
| KV cache | FP8 scaling metadata; intended for an FP8 runtime cache |
| Vision tower | BF16 |
| MTP head | BF16 |
| Embeddings, norms, and control components | BF16 |

Quantization used **NVIDIA ModelOpt 0.46.0**. The 5.0-bit target applies to the
compatible quantized language weights, not the BF16 components or runtime KV
cache. Exact module assignments are recorded in `hf_quant_config.json` and
`config.json`.

The complete repository occupies approximately **20.39 GB (18.99 GiB)**.

## Calibration

Calibration used **1,024 fully populated 512-token samples** from the
deterministic version-6 corpus (seed `42`). Its token distribution was:

| Calibration source | Token share |
|---|---:|
| [Nemotron SFT Instruction Following Chat v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | 20% |
| [Nemotron Science v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | 8% |
| [Nemotron Competitive Programming v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | 12% |
| [Nemotron SFT Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) — function calling | 20% |
| [Nemotron SFT Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) — web search | 5% |
| [Nemotron Math v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | 15% |
| [Nemotron SFT SWE v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) — agentless SWE | 10% |
| [Nemotron SFT SWE v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) — OpenHands SWE | 10% |

## Quality Benchmarks

The first row contains author-reported results for this checkpoint. The
comparison rows use the benchmark table published for
[acyildirimer/Qwen3.8-27B-NVFP4](https://huggingface.co/acyildirimer/Qwen3.8-27B-NVFP4).

Thinking was disabled during these quality benchmark runs to keep the evaluation
conditions consistent and avoid mixing reasoning-mode differences into the
checkpoint comparison.

| Checkpoint | Quantization | Backend | MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | 4-task mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **acyildirimer/Qwen3.8-27B-AutoQuant-NVFP4-FP8-5bit** | **ModelOpt NVFP4 + FP8, 5-bit target** | **vLLM** | **84.43** | **96.25** | **94.07** | **96.82** | **93.29** | **92.89** |
| [acyildirimer/Qwen3.8-27B-NVFP4](https://huggingface.co/acyildirimer/Qwen3.8-27B-NVFP4) | ModelOpt NVFP4 + FP8 | vLLM | 85.1 | 96.8 | 94.1 | 96.9 | 92.1 | 93.2 |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | BF16 | llama.cpp (partial offload) | 85.3 | 96.8 | 94.3 | 97.4 | 93.9 | 93.5 |
| [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | Dynamic V3 NVFP4 | vLLM | 84.3 | 96.9 | 94.3 | 97.1 | 89.6 | 93.2 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q8_0 | llama.cpp | 85.2 | 96.8 | 94.4 | 97.4 | 94.5 | 93.5 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q6_K | llama.cpp | 85.3 | 96.7 | 94.3 | 97.5 | 94.5 | 93.5 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL | llama.cpp | 85.1 | 96.6 | 94.4 | 97.3 | 93.9 | 93.4 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q4_K_M | llama.cpp | 85.0 | 96.8 | 94.3 | 97.1 | 92.7 | 93.3 |

`4-task mean` is the arithmetic mean of MMLU, ARC-Challenge, HellaSwag, and
GSM8K. HumanEval is reported separately and is not included in that aggregate.
The first-row MMLU and HellaSwag results use their complete evaluation sets:
MMLU is 5-shot accuracy over 14,042 questions (**11,856 correct**), and
HellaSwag is 10-shot accuracy over 10,042 questions (**9,447 correct**). Both
were evaluated through vLLM chat completions with thinking disabled,
temperature `0`, and strict generated-letter scoring. These are generative
accuracy results, not log-likelihood leaderboard scores.

The first-row results are author-reported. The remaining comparison results
originate from the public
[`witcheer/rtx-5090-benchmarks`](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks)
dataset. The serving stacks differ, so the comparison is indicative rather than
a controlled quantization ablation.

## Tool-Calling Evaluation

Evaluated with `tool-eval-bench` v2.2.0 through vLLM at temperature `0`, seed
`42`, across all 69 scenarios:

| Points | Normalized score | Outcomes |
|---:|---:|---:|
| **127 / 138** | **92 / 100** | **60 passed, 7 partial, 2 failed** |

One failure involved prompt-injection resistance. Tool-enabled deployments
should enforce independent authorization and defenses for untrusted tool, file,
and web content.

## vLLM Usage

This checkpoint was designed for headless deployment on a dual-GPU system with
**32 GB of total VRAM**, while retaining the full native context window and
vision support. Both memory-conscious configurations below are set to serve one
request at a time.

A vLLM runtime with ModelOpt mixed NVFP4/FP8 support is required. This
checkpoint was smoke-tested with vLLM 0.28.0.

```bash
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=1 \
vllm serve acyildirimer/Qwen3.8-27B-AutoQuant-NVFP4-FP8-5bit \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization modelopt \
    --tensor-parallel-size 2 \
    --safetensors-load-strategy lazy \
    --kv-cache-dtype fp8 \
    --max-model-len auto \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.97 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --performance-mode interactivity \
    --attention-backend auto \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --generation-config auto \
    --override-generation-config \
      '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs \
      '{"preserve_thinking":true,"reasoning_effort":"xhigh"}'
```

Vision support is enabled by leaving out `--language-model-only`. This example
also disables NCCL P2P communication and vLLM's custom all-reduce path for a
dual-GPU system without reliable peer-to-peer connectivity.

The SGLang configuration below provides an equivalent multimodal setup.

## SGLang Usage

The equivalent SGLang configuration is:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang serve \
    --trust-remote-code \
    --model-path acyildirimer/Qwen3.8-27B-AutoQuant-NVFP4-FP8-5bit \
    --tp 2 \
    --disable-custom-all-reduce \
    --enable-multimodal \
    --mem-fraction-static 0.93 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 2048 \
    --max-running-requests 1 \
    --cuda-graph-bs-decode 1 \
    --disable-prefill-cuda-graph \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size 4 \
    --mamba-ssm-dtype float32 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs \
      '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
    --host 0.0.0.0 \
    --port 8000
```

This configuration intentionally disables NCCL P2P communication and SGLang's
custom all-reduce path for dual-GPU systems without reliable peer-to-peer
connectivity.

## Limitations

This is a lossy post-training quantization. Results can vary with the evaluation
protocol, runtime, GPU architecture, context length, and workload. The model also
inherits the capabilities and limitations of the upstream checkpoint.

Refer to the
[Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the
authoritative architecture description, generation settings, vision
preprocessing, and upstream limitations.

## Acknowledgements

This checkpoint retains the upstream **Apache 2.0** license and was quantized
with [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer).
