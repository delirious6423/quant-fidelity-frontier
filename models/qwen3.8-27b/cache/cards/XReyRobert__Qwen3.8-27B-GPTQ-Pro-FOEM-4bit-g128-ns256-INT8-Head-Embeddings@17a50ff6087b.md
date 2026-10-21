---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
  - qwen3.8
  - gptq
  - gptq-pro
  - marlin
  - int8
  - vllm
  - speculative-decoding
  - long-context
  - reasoning
  - tool-use
  - function-calling
  - terminal-bench
---

![Qwen3.8-27B GPTQ-Pro serving metrics on one RTX 3090](./assets/qwen38-27b-gptqpro-ampere-rtx3090-vllm-qwen-logo.png)

# Qwen3.8-27B GPTQ-Pro Hybrid INT8

> [!IMPORTANT]
> **Patched vLLM 0.27.1 is required.** Stock vLLM 0.27.1 does not correctly
> route this checkpoint's compressed-tensors INT8 token embedding when the
> transformer body uses GPTQ. Build and use the included `Containerfile.vllm`
> compatibility image before serving the model. The patch changes loader and
> kernel routing only; it does not modify the model weights, sampling, or
> generation semantics. See [Runtime compatibility](#runtime-compatibility).

This checkpoint is a memory-optimized derivative of
[`XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256`](https://huggingface.co/XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256).
It keeps the GPTQ-Pro FOEM 4-bit transformer body and further quantizes the
large token embedding and output projection tensors to INT8.

The artifact was built for long-context inference on a single 24 GB GPU. It
preserves the vision encoder, MTP layer, and normalization tensors from the
source checkpoint, so the same weights can be used for text-only, vision, and
MTP speculative-decoding profiles.

The GPTQ-Pro body was produced from source revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Provenance and credits

- The base model is from the [Qwen team](https://huggingface.co/Qwen).
- The GPTQ-Pro transformer body was produced with
  [GPTQModel](https://github.com/ModelCloud/GPTQModel), using the FOEM recipe
  documented below.
- The INT8 `lm_head` and token-embedding transformation, serving approach,
  and original `qwen3_5-embed-quant.patch` are adapted from
  [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)
  at pinned revision
  [`e681a8f6b190cca1f86d58b8b333b64d014994d7`](https://github.com/syv-ai/qwen38-27b-rtx3090/commit/e681a8f6b190cca1f86d58b8b333b64d014994d7).
- `qwen38-gptq-hybrid-embedding.patch` is the additional adaptation for this
  checkpoint. It routes the compressed-tensors INT8 embedding through
  vLLM's quantized-embedding implementation while the transformer body uses
  GPTQ.

The redistributed upstream patch and its Apache-2.0 license are identified in
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

## Quantization recipe

| Component | Stored format | Group size | Notes |
|---|---:|---:|---|
| Transformer body | GPTQ-Pro FOEM W4A16, symmetric | 128 | `desc_act=false`, true sequential |
| Token embeddings | packed W8A16, symmetric | 128 | compressed-tensors embedding layout |
| `lm_head` | GPTQ W8A16, symmetric | 128 | native GPTQ packed layout |
| Vision encoder | preserved BF16 | - | not quantized |
| MTP layer | preserved BF16 | - | supports vLLM MTP speculative decoding |
| Norms | preserved BF16 | - | not quantized |

The GPTQ-Pro body used 256 calibration samples of length 2048 from a mixed
code/reasoning JSONL calibration set. The recorded body recipe uses
GPTQModel `6.1.0-dev`, activation-weighted MSE, act-group-aware quantization,
FOEM `alpha=0.25`, `beta=0.2`, and a 0.5% RTN fallback threshold.

The second-stage INT8 conversion report records full stored-weight
reconstruction errors against the source checkpoint:

| Tensor | Relative L2 error |
|---|---:|
| Token embeddings | 0.0065123175 |
| `lm_head` | 0.0069414420 |

The complete machine-readable settings are included in
`quantize_config.json`, `config.json`, and `quantization_report.json`.

## Runtime compatibility

**Important:** this hybrid embedding layout requires a narrow vLLM
compatibility patch. Stock vLLM 0.27.1 does not route the
`hybrid_embedding_w8a16` declaration to
`CompressedTensorsEmbeddingWNA16Int` when the rest of the checkpoint uses
GPTQ.

This repository includes:

- `Containerfile.vllm`
- `patches/qwen3_5-embed-quant.patch`
- `patches/qwen38-gptq-hybrid-embedding.patch`
- `patches/v0271-kimi-k3-warmup-model-gate.patch`

The first two patches provide the Qwen3.5 embedding quantization path and the
hybrid GPTQ/INT8 routing. The warmup patch is included to reproduce the exact
validated container image; it does not change model weights or generation
semantics.

Build the compatibility image before serving:

```bash
podman build -f Containerfile.vllm -t qwen38-gptq-hybrid-vllm:0.27.1 .
```

## Text-only serving

The following profile was validated with vLLM 0.27.1 on one RTX 3090 24 GB:

```bash
vllm serve XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256-INT8-Head-Embeddings \
  --served-model-name qwen3.8-27b-gptq-pro-hybrid-int8-g128-ctx210k \
  --language-model-only \
  --dtype float16 \
  --quantization gptq_marlin \
  --max-model-len 210000 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8_e5m2 \
  --kv-cache-memory 7300000000 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-cache-mode align \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --compilation-config '{"max_cudagraph_capture_size":16,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --gpu-memory-utilization 0.985 \
  --trust-remote-code
```

This profile intentionally does not use `--enforce-eager`.

## MTP2 serving

The same checkpoint was also validated with two speculative MTP tokens at a
170K context window. On the same 24 GB GPU, this requires a tighter CUDA graph
profile and a 6.7 GB explicit KV allocation:

```bash
vllm serve XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256-INT8-Head-Embeddings \
  --served-model-name qwen3.8-27b-gptq-pro-hybrid-int8-g128-ctx170k-mtp2 \
  --language-model-only \
  --dtype float16 \
  --quantization gptq_marlin \
  --max-model-len 170000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8_e5m2 \
  --kv-cache-memory 6700000000 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --compilation-config '{"max_cudagraph_capture_size":4,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --gpu-memory-utilization 0.985 \
  --trust-remote-code
```

Observed MTP acceptance is workload-dependent. A long single-session coding
run reached approximately 44 output tokens/s over a 15.5-second sample, with
85.8% recent draft-token acceptance and about 70.6% cumulative acceptance.
Treat these numbers as an operational observation, not a standardized
benchmark result.

## Vision profile

Vision weights are present. A separate vLLM profile was validated at 182K
context with one image per prompt, video disabled, and a 6.0 GB FP8 E5M2 KV
allocation. Do not pass `--language-model-only` when serving this profile.

## Sampling and thinking control

Use the source model's recommended sampling parameters:

| Mode | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
|---|---:|---:|---:|---:|---:|---:|
| Thinking | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Instruct / non-thinking | `0.7` | `0.8` | `20` | `0.0` | `1.5` | `1.0` |

Qwen3.8 supports `reasoning_effort=low`, `medium`, or `xhigh`; `xhigh` is the
source-model default. For multi-turn agentic work, preserve previous thinking
blocks through the chat template:

```json
{"enable_thinking": true, "preserve_thinking": true}
```

Pass that object as `chat_template_kwargs`. For direct non-thinking replies,
set `enable_thinking=false` and use the non-thinking sampling row above.

For the validated 210K deployment, the long-horizon benchmark profile reserves
131,250 tokens for input/history, 52,500 for reasoning, and 26,250 for the
final response. These are deployment budgets, not limits stored in the model
weights.

## Agentic validation

### Terminal-Bench 2.0 Smoke24 effort comparison

The 210K text profile completed one full pass over the fixed 24-task Smoke24
subset for each supported reasoning effort. The checkpoint, serving profile,
sampling, task order, and resource limits were held constant; only
`reasoning_effort` changed.

| Effort | Score | Campaign elapsed | Harbor-visible output | vLLM output | Output / success | Parser warnings (trials) | Agent timeouts | Decode | Prefill | Prefix cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `low` | **16/24 (66.7%)** | 372.5 min | 680,825 | 749,287 | 42,552 | 31 (6) | 4 | 44.9 tok/s | 1,123 tok/s | 72.8% |
| `medium` | **17/24 (70.8%)** | 369.2 min | 668,112 | 789,519 | 39,301 | 44 (5) | 4 | 45.4 tok/s | 1,089 tok/s | 80.4% |
| `xhigh` | **17/24 (70.8%)** | 339.2 min | 638,308 | 726,355 | 37,548 | 26 (10) | 3 | 45.4 tok/s | 1,068 tok/s | 80.2% |

These runs used Terminal-Bench 2.0, Terminus-2, `parser=json`, one RTX 3090,
one concurrent trial, 32 CPU cores, 48 GiB RAM, and a 30-minute timeout per
task. Sampling was `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`,
presence penalty 0, and repetition penalty 1. Thinking and thinking
preservation were enabled. Each task received 131,250 input/history tokens,
52,500 reasoning tokens, and 26,250 final-response tokens within the 210K
deployment envelope.

`medium` and `xhigh` tied one task above `low`, while `xhigh` used the least
Harbor-visible output and completed fastest in this pass. This is a 24-task
subset with one pass per effort, so the one-task difference is not
statistically decisive. Timeouts and parser-related failures remain measured
benchmark outcomes and were not retried.

Detailed artifacts:

- [`smoke24_effort_summary_20260830.csv`](./benchmarks/terminal-bench-2.0/smoke24_effort_summary_20260830.csv)
- [`smoke24_effort_trials_20260830.csv`](./benchmarks/terminal-bench-2.0/smoke24_effort_trials_20260830.csv)
- [`smoke24_effort_report_20260830.md`](./benchmarks/terminal-bench-2.0/smoke24_effort_report_20260830.md)
- [`smoke24_task_list_20260616.md`](./benchmarks/terminal-bench-2.0/smoke24_task_list_20260616.md)

### Terminal-Bench 2.0 sanity check

The checkpoint was exercised with Terminus 2 on three tasks
(`openssl-selfsigned-cert`, `headless-terminal`, and
`count-dataset-tokens`) using one RTX 3090, one concurrent trial, 32 CPU
cores, 48 GiB RAM, and a 30-minute task timeout.

| Profile | Effort | Verifier reward | Run wall | Visible output | vLLM output | Parser warnings | Decode | Prefill | Agent timeouts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 210K text | `low` | 3/3 | 36.8 min | 27,509 | 103,487 | 0 | 45.1 tok/s | 1,161 tok/s | 0 |
| 210K text | `xhigh` | 1/3 | 52.8 min | 116,305 | 120,260 | 3 | 45.6 tok/s | 1,174 tok/s | 1 |
| 170K MTP2 | `low` | 3/3 | 26.8 min | 25,660 | 42,315 | 1 | 42.3 tok/s | 1,133 tok/s | 0 |
| 170K MTP2 | `xhigh` | 3/3 | 35.7 min | 65,400 | 65,900 | 0 | 39.8 tok/s | 1,143 tok/s | 0 |

The 210K xhigh timeout occurred after active generation and is retained as a
benchmark performance outcome. Parser warnings indicate extra or discarded
text around tool actions; they do not automatically imply an incorrect tool
call. This three-task check validates the serving and agent path but is too
small for broad comparative quality claims.

## Validation status

- vLLM 0.27.1 compatibility image starts successfully without eager mode.
- OpenAI-compatible `/v1/models` and chat completions were validated.
- Text-only inference was exercised near the 210K context limit.
- Prefix caching, Qwen reasoning parsing, tool-call parsing, and MTP2 were
  exercised in the deployed profiles.
- The 182K vision profile completed an image request.

The published agentic validation is limited to the Smoke24 subset and the
three-task sanity profiles above; no broad standardized accuracy suite has
been run on this derivative. Context limits are deployment-specific and depend
on GPU memory, KV-cache dtype, CUDA-graph settings, concurrency, and vLLM
version.

## Limitations

- The custom embedding path currently requires the included vLLM patch.
- FP8 KV cache can introduce additional numerical loss independently of weight
  quantization.
- MTP speculative decoding changes memory requirements and may not improve
  every workload.
- The checkpoint inherits the capabilities, risks, and intended-use guidance
  of the original Qwen3.8-27B model.
