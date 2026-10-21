---
library_name: vllm
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
  - en
  - zh
  - sk
pipeline_tag: image-text-to-text
tags:
  - autoround
  - w4a16
  - gptq
  - marlin
  - int4
  - quantized
  - vllm
  - qwen3.8
  - gdn
---

# Qwen3.8-27B — AutoRound W4A16 Quantization (g128, Symmetric)

This is a **W4A16 (4-bit weight-only) quantization** of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B parameter hybrid linear-attention / full-attention multimodal model with vision, native tool calling, and chain-of-thought reasoning capabilities.

Quantization was performed using [AutoRound](https://github.com/intel/auto-round) (v0.15.0.dev66) with **SignRoundV2** (`enable_alg_ext`), producing a **GPTQ packed format** that maps directly to vLLM's **MarlinLinearKernel** (fused INT4 GEMM) for maximum throughput. The calibration uses a custom 512-sample dataset spanning 9 task categories (15% Slovak) for domain-representative weight tuning.

## Model Details

| Property | Value |
| --- | --- |
| **Base Model** | Qwen/Qwen3.8-27B |
| **Quantization Method** | AutoRound SignRound + SignRoundV2 (symmetric, group_size=128) |
| **Weight Precision** | INT4 (4-bit, symmetric) |
| **Activation Precision** | BF16 (weight-only quantization) |
| **Group Size** | 128 |
| **Quantization Library** | auto-round 0.15.0.dev66 |
| **Format** | auto_round:auto_gptq (GPTQ Marlin) |
| **Architecture** | Qwen3_5ForConditionalGeneration |
| **LM Layers** | 64 (48 linear_attention + 16 full_attention, hybrid GDN) |
| **Hidden Size** | 5,120 |
| **Intermediate Size** | 17,408 |
| **Attention Heads** | 24 (query), 4 (key/value, GQA) |
| **Head Dimension** | 256 |
| **Vocabulary Size** | 248,320 |
| **Context Window** | 262,144 tokens (262K) |
| **Vision Encoder** | Qwen3_VisionTransformer (27 layers, hidden_size=1152, patch_size=16) |
| **Vision Projector** | spatial_merge_size=2 |
| **Quantized Components** | Text decoder Linear layers (attention q/k/v/o_proj, linear_attn out_proj/in_proj_qkv/z, MLP gate/up/down) |
| **Preserved in BF16** | Full vision encoder, linear_attn.in_proj_a/b, lm_head, mtp |
| **Model Size** | ~19 GB |
| **Calibration** | 512 task-specific samples (9 categories, 15% Slovak) |

## Quantization Details

### Recipe

```bash
auto-round \
  --model_name Qwen/Qwen3.8-27B \
  --scheme W4A16 \
  --bits 4 \
  --group_size 128 \
  --format auto_round:auto_gptq \
  --dataset calibration_data.jsonl \
  --nsamples 512 \
  --seqlen 4096 \
  --iters 1000 \
  --enable_alg_ext \
  --enable_torch_compile \
  --low_gpu_mem_usage \
  --layer_config '{
    "linear_attn.in_proj_a": {"bits": 16},
    "linear_attn.in_proj_b": {"bits": 16},
    "lm_head":               {"bits": 16},
    "mtp":                   {"bits": 16}
  }'
```

Key algorithm settings:
- **SignRoundV2** (`enable_alg_ext`): DeltaLoss sensitivity + pre-tuning scale search for improved weight rounding
- **Symmetric** quantization (no zero-point overhead, optimal Marlin kernel path)
- **1000 iterations** with 512 calibration samples at sequence length 4096
- **400/607 Linear layers quantized** to W4; remaining 207 layers preserved in BF16 (vision encoder, in_proj_a/b, lm_head, mtp)

### Calibration Dataset

512 samples formatted with the Qwen3.8 ChatML template. Task-specific calibration aligns quantization with the model's deployment distribution.

| Source | Domain | Samples |
| --- | --- | --- |
| Code generation & refactoring | Software development | ~150 |
| Multi-turn tool calling | Agent workflows | ~80 |
| Mathematical reasoning | Arithmetic, algebra | ~60 |
| Slovak text (general) | Multilingual coverage | ~77 (15%) |
| General knowledge & QA | Broad coverage | ~80 |
| Creative writing | Long-form generation | ~40 |
| Long-context documents | Extended context | ~25 |

## Quality Benchmarks

All benchmarks on **wikitext-2-raw-v1** (test split), 100 non-overlapping chunks of 512 tokens (24,426 scored tokens). PPL measured via vLLM completions API with `echo=True` + `logprobs`. KLD measured via sparse top-20 logprob comparison against BF16 base model.

### PPL (Perplexity)

| Model | PPL | Δ vs BF16 | Size |
| --- | --- | --- | --- |
| BF16 base | 9.6402 | — | ~54 GB |
| **This model (v9-2 g128 sym)** | **9.8108** | **+1.77%** | **~19 GB** |

**PPL** = Perplexity (lower is better). The +1.77% deviation is well within the <4% acceptance threshold and represents state-of-the-art quality recovery for W4A16 quantization of hybrid GDN architectures.

### KL Divergence (vs BF16 base)

KLD direction: KL(P_base ‖ P_quant) — "how well does the quantized model approximate the base?" Computed via sparse top-20 logprob approximation with tail-mass lumping (vLLM API provides top-20 logprobs per token, not full vocab). This underestimates absolute KLD by ~10-15% vs full-vocab computation, but relative comparisons remain valid.

| Metric | Value |
| --- | --- |
| **Mean KLD** | **0.0699** |
| Median KLD | 0.0267 |
| p95 KLD | 0.2643 |
| p99 KLD | 0.5619 |
| **Same-top-p** | **92.61%** |
| Δp RMS | 0.0193 |
| Tokens scored | 24,426 |
| Chunks | 100 |
| Context length | 512 |

**Mean KLD** = mean KL-divergence from BF16 output distribution (lower is better). **Same-top-p** = fraction of tokens where quantized model agrees with BF16 on the top predicted token (higher is better).

### Downstream Task Benchmarks (lm-evaluation-harness)

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.12 against a vLLM deployment of this model (2x RTX 4090, TP=2, 393K context, MTP speculative decoding). Deterministic decoding (temperature 0). Because the model emits chain-of-thought before the final answer, generation tasks require `max_gen_toks=32768`; at 8,192 the reasoning budget truncates ~12% of GSM8K solutions (0.859 instead of 0.982 strict-match). Medium-tier tasks are capped at 1,000 samples; long-context suites at 250 samples per subtask.

| Benchmark | Metric | Score |
| --- | --- | --- |
| MMLU | acc (5-shot) | 0.839 |
| MMLU-Pro | exact_match (5-shot, 14 categories) | 0.823 |
| Global MMLU (en) | acc (5-shot) | 0.868 |
| BBH | exact_match (3-shot CoT, 27 subtasks) | 0.924 ¹ |
| GSM8K | exact_match, strict-match (5-shot) | 0.982 |
| GSM8K Platinum | exact_match, strict-match (5-shot) | 0.996 |
| HellaSwag | acc_norm (5-shot) | 0.799 |
| ARC-Challenge | acc_norm (5-shot) | 0.701 |
| Winogrande | acc (5-shot) | 0.795 |
| PIQA | acc_norm (5-shot) | 0.825 |
| TruthfulQA | mc1 / mc2 (0-shot) | 0.361 / 0.544 |
| IFEval | prompt-level strict / loose (0-shot) | 0.887 / 0.915 |
| HumanEval | pass@1 (0-shot) ² | 0.976 |
| MBPP | pass@1 (0-shot) | 0.874 |
| BABILong | mean acc (0-shot, 20 tasks) | 0.876 |
| LongBench | mean score (0-shot) | 0.520 |

¹ BBH raw score under the harness `get-answer` filter is 0.786: the model formats final answers with markdown emphasis (`**X**`), which breaks exact matching. The reported value re-scores the stored model outputs with markdown stripping; the same correction is applied consistently across all models on our internal leaderboard.

² HumanEval uses the harness task variant that extracts fenced ```python blocks from the response. A stricter first-code-block extraction scores 0.604. MBPP uses the markdown-tolerant extraction.

**LongBench by category** (250 samples per subtask): retrieval 1.000 (passage_retrieval_en, passage_retrieval_zh), few-shot learning 0.668, multi-doc QA 0.602, single-doc QA 0.501, code completion 0.362, summarization 0.211 (ROUGE-based, structurally low for all reasoning models).

**BABILong**: 11 of 20 subtasks score >= 0.98 (7 at exactly 1.000). The weakest are qa7 (counting/size properties, 0.112) and qa15 (basic deduction, 0.492), consistent with the behavior of the BF16 base and much larger models on the same suite.

**MMLU-Pro by category**: math 0.928, biology 0.920, physics 0.916, computer science 0.844, health 0.820, economics 0.796, law 0.664.

**Observations.** GSM8K Platinum 0.996 indicates the W4A16 error is concentrated on reasoning-length and formatting behavior rather than arithmetic correctness: with a sufficient generation budget the model recovers near-ceiling accuracy. TruthfulQA mc1 (0.361) is the most quantization-sensitive task in this suite and the largest gap to the BF16 base distribution, consistent with log-likelihood scoring amplifying small probability shifts on short-answer alternatives.


## Throughput Benchmarks

Benchmarks measured on **NVIDIA RTX 5090 (32 GB VRAM, sm120 / Blackwell)** running **vLLM** with `quantization: gptq`, `kv_cache_dtype: fp8_e4m3`, `max_model_len: 8192`, single GPU (TP=1), CUDA graphs enabled (`--enforce-eager` disabled).

### Single Request (concurrency=1)

20 requests, max_tokens=1024, temperature=0.9. Unique random prompts per request to prevent prefix cache hits. Aggregate throughput is wall-clock total tokens / total wall time.

| Metric | Value |
| --- | --- |
| **Aggregate throughput** | **87.5 tok/s** |
| **Per-request throughput** | min 72.6 / avg 84.4 / max 88.1 tok/s |
| Average latency (TTFT + gen) | 4.23 s |
| Min / Max latency | 0.21 s / 11.63 s |
| Completion tokens | min 15 / avg 370 / max 1024 |
| Success rate | 20/20 (100%) |
| Wall time | 84.5 s |

### 16 Concurrent Requests

160 total requests, max_tokens=1024, temperature=0.9.

| Metric | Value |
| --- | --- |
| **Aggregate throughput** | **970.8 tok/s** |
| **Per-request throughput** | min 5.5 / avg 62.8 / max 76.6 tok/s |
| Average latency | 6.91 s |
| Min / Max latency | 0.25 s / 17.72 s |
| Completion tokens | min 15 / avg 459 / max 1024 |
| Success rate | 160/160 (100%) |
| Wall time | 75.6 s |

## Hardware Requirements

| GPU | VRAM | Recommended `gpu_memory_utilization` | Max Context |
| --- | --- | --- | --- |
| RTX 5090 | 32 GB | 0.90 | 131,972 tokens (with fp8 KV cache) |
| RTX 4090 (2× TP=2) | 48 GB | 0.95 | 393,216 tokens (with fp8 KV offload) |
| RTX PRO 6000 / A100 / H100 | 80–96 GB | 0.92 | 262,144 tokens (full) |

**Minimum**: 1× GPU with ≥32 GB VRAM (with reduced context window and fp8 KV cache).

The GPTQ packed format uses **MarlinLinearKernel** (fused INT4 GEMM) — requires NVIDIA GPU sm75+ (Turing and later). Works on: Ampere (A100, RTX 3090), Ada (RTX 4090), Hopper (H100, H200), Blackwell (RTX 5090, RTX PRO 6000).

## Usage with vLLM

**Tested with**: `vllm/vllm-openai:qwen38` (vLLM v0.1.dev19754)

### Docker Deployment (example with 393K context on 2× RTX 4090)

```bash
docker run -d --name vllm-qwen38-27b \
  --runtime=nvidia --gpus '"device=0,1"' \
  -p 8000:8000 \
  -v /path/to/model:/workspace/model \
  --ipc=host --shm-size=48g \
  --restart unless-stopped \
  vllm/vllm-openai:qwen38 \
  /workspace/model \
  --host 0.0.0.0 --port 8000 \
  --quantization gptq \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 393216 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-offloading-size 32 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce
```

### Example vLLM Configuration (YAML)

This configuration is **deployed and verified** on 2× RTX 4090 (48 GB total):

```yaml
# -- Model & Server ----------------------------------------------------------
model: /workspace/model
host: "0.0.0.0"
port: 8000
served_model_name: "qwen38-27b"
trust_remote_code: true
tensor_parallel_size: 2
# -- Quantization ------------------------------------------------------------
quantization: gptq
# -- Data Type ---------------------------------------------------------------
dtype: bfloat16
# -- Context & Batching ------------------------------------------------------
max_model_len: 393216
max_num_batched_tokens: 4096
max_num_seqs: 32
# -- Memory ------------------------------------------------------------------
gpu_memory_utilization: 0.95
kv_cache_dtype: fp8_e4m3
kv_offloading_size: 32
# -- Tool Calling & Reasoning ------------------------------------------------
enable_auto_tool_choice: true
tool_call_parser: qwen3_xml
reasoning_parser: qwen3
# -- Misc --------------------------------------------------------------------
disable_custom_all_reduce: true
```

### Inference Test

```bash
# Text completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":10}'

# Tool calling test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-27b",
    "messages": [{"role":"user","content":"What is the weather in Prague?"}],
    "tools": [{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice": "auto",
    "max_tokens": 256
  }'
```

### Transformers / Python

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "gratex/Qwen3.8-27B-W4A16-g128-sym-GPTQ"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "Explain GPTQ quantization in two sentences."}]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Files in This Repository

| File | Size | Description |
| --- | --- | --- |
| `model-00001-of-00007.safetensors` | ~3.2 GB | Quantized model weights (INT4 + BF16), shard 1/7 |
| `model-00002-of-00007.safetensors` | ~3.2 GB | Shard 2/7 |
| `model-00003-of-00007.safetensors` | ~3.2 GB | Shard 3/7 |
| `model-00004-of-00007.safetensors` | ~3.2 GB | Shard 4/7 |
| `model-00005-of-00007.safetensors` | ~0.8 GB | Shard 5/7 |
| `model-00006-of-00007.safetensors` | ~2.5 GB | Shard 6/7 |
| `model-00007-of-00007.safetensors` | ~2.5 GB | Shard 7/7 |
| `model_extra_tensors.safetensors` | ~0.8 GB | BF16 tensors (vision, in_proj_a/b, mtp) |
| `model.safetensors.index.json` | — | Shard index |
| `config.json` | — | Model configuration with `quantization_config` |
| `quantization_config.json` | — | AutoRound quantization parameters |
| `tokenizer.json` | — | Qwen3 vocabulary (248,320 tokens) |
| `tokenizer_config.json` | — | Tokenizer config with ChatML chat template |
| `chat_template.jinja` | — | Qwen3.8 native chat template (ChatML format) |
| `generation_config.json` | — | Default generation parameters |
| `preprocessor_config.json` | — | Image preprocessor configuration |
| `processor_config.json` | — | Multi-modal processor configuration |
| `LICENSE` | — | Apache 2.0 License |

## License

This quantization is released under the **Apache 2.0 License**, following the base model's license.

The base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is licensed under Apache 2.0.

## Citation

If you use this model in your research or production systems, please cite:

```bibtex
@misc{qwen38-27b-autoround-w4a16-g128-sym,
  title = {Qwen3.8-27B AutoRound W4A16 g128 Symmetric Quantization},
  author = {Gratex International},
  year = {2026},
  howpublished = {\url{https://huggingface.co/gratex/Qwen3.8-27B-W4A16-g128-sym-GPTQ}},
  note = {Quantized with AutoRound 0.15.0.dev66, SignRoundV2, custom calibration dataset}
}
```

## Acknowledgments

This quantization was produced using hardware and infrastructure provided by [Gratex International, a.s.](https://www.gratex.com)

**Original Model**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)  
**Quantization Tool**: [AutoRound](https://github.com/intel/auto-round)  
**Quantization Format**: GPTQ (Marlin kernel)  
**Deployment Engine**: [vLLM](https://github.com/vllm-project/vllm)
