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
  - multimodal
  - vllm
  - long-context
  - reasoning
  - tool-use
  - function-calling
---

# Qwen3.8-27B GPTQ-Pro FOEM 4-bit

This is a GPTQ-Pro FOEM quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). It targets
efficient single-GPU inference while preserving the base model's large token
embedding, output projection, vision, MTP, and normalization tensors outside
the 4-bit transformer-body path.

The checkpoint retains the base model's native 262,144-token position limit.
The maximum practical serving context is deployment-specific and depends on
GPU memory, KV-cache precision, concurrency, CUDA-graph capture, and whether
vision or speculative MTP decoding is enabled.

The quantization was produced from source revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Quantization recipe

| Setting | Value |
|---|---|
| Transformer body | GPTQ-Pro FOEM W4A16 |
| Weight bits | 4 |
| Group size | 128 |
| Symmetric | Yes |
| Activation ordering | Disabled (`desc_act=false`) |
| Sequential quantization | Enabled |
| Calibration samples | 256 |
| Calibration sequence length | 2048 |
| Calibration mix | Code and reasoning JSONL |
| Quantizer | GPTQModel `6.1.0-dev` |
| Pack format | GPTQ, INT32 packing |

The recorded recipe uses activation-weighted MSE, act-group-aware
quantization, FOEM `alpha=0.25` and `beta=0.2`, plus a 0.5% RTN fallback with
MSE smoothing. Full machine-readable settings are available in
`quantize_config.json` and `config.json`.

## Preserved tensors

The following tensor families were excluded from the 4-bit GPTQ body and kept
in their original precision:

- token embeddings
- `lm_head`
- vision encoder tensors
- MTP tensors
- normalization tensors

The repository includes `model-preserved-skips.safetensors` as an auxiliary
record of the 15 explicitly preserved tensors. The indexed checkpoint contains
2,399 weights totaling 19,559,449,368 bytes (approximately 18.22 GiB).

## vLLM serving

The following text-only profile was validated on one RTX 3090 24 GB. The
runtime was the local residency image
`localhost/vllm-openai-nightly-7a1eb8ac2-sleepwake-local:20260617`, based on
the pinned upstream vLLM nightly commit
`7a1eb8ac2ec4ea69338c51dc7afd4b15010abfa8`. Its local changes implement
sleep/wake residency control; they do not alter this checkpoint's
quantization format.

```bash
vllm serve XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256 \
  --served-model-name qwen3.8-27b-gptq-pro-foem-4bit-g128-ns256-ctx110k \
  --language-model-only \
  --dtype float16 \
  --quantization gptq_marlin \
  --max-model-len 110592 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-cudagraph-capture-size 32 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code
```

This validated profile does not use `--enforce-eager`. The 110,592-token
limit is a single-RTX-3090 deployment result, not a limitation encoded in the
checkpoint.

Vision and MTP weights are present, but enabling either feature changes the
memory budget and requires a separately sized serving profile. Remove
`--language-model-only` for vision use.

## Sampling and thinking control

Use the source model's recommended sampling parameters:

| Mode | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
|---|---:|---:|---:|---:|---:|---:|
| Thinking | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Instruct / non-thinking | `0.7` | `0.8` | `20` | `0.0` | `1.5` | `1.0` |

Qwen3.8 supports `reasoning_effort=low`, `medium`, or `xhigh`; `xhigh` is the
source-model default. Thinking is enabled by default. For agentic sessions,
retain prior reasoning blocks with:

```json
{"enable_thinking": true, "preserve_thinking": true}
```

Pass that object as `chat_template_kwargs`. For direct non-thinking replies,
set `enable_thinking=false` and use the non-thinking sampling row above.

Large `max_tokens` values are intentional for long-horizon agentic tasks. The
source model recommends separate reasoning and final-response headroom when
the serving stack supports those limits. Keep input, retained reasoning, and
new output within the deployment's actual `max_model_len`.

## Longer-context hybrid variant

For single-24-GB-GPU deployments requiring more KV-cache capacity, see
[`XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256-INT8-Head-Embeddings`](https://huggingface.co/XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256-INT8-Head-Embeddings).
That derivative keeps this 4-bit GPTQ-Pro transformer body and additionally
quantizes the token embeddings and `lm_head` to group-size-128 INT8. It was
validated at 210K text-only context, 182K with vision, and 170K with MTP2 on
one RTX 3090 24 GB. It requires the narrow vLLM compatibility patch included
in that repository.

The hybrid derivative's published Terminal-Bench 2.0 Smoke24 campaign scored
`16/24` with `reasoning_effort=low` and `17/24` with both `medium` and
`xhigh`, using one pass per effort on its 210K text profile. See the
[hybrid model card](https://huggingface.co/XReyRobert/Qwen3.8-27B-GPTQ-Pro-FOEM-4bit-g128-ns256-INT8-Head-Embeddings#terminal-bench-20-smoke24-effort-comparison)
for the exact conditions and per-task artifacts. These results belong to the
hybrid INT8 derivative and must not be attributed to this non-hybrid
checkpoint.

## Validation status

- The checkpoint loads with GPTQ-Marlin in the pinned vLLM nightly runtime.
- OpenAI-compatible chat inference, Qwen reasoning parsing, tool-call parsing,
  prefix caching, and residency sleep/wake were exercised in deployment.
- Text-only serving was validated with `max_model_len=110592` on one RTX 3090.
- No standardized quality benchmark has been run specifically on this
  non-hybrid checkpoint. Do not transfer results from the hybrid INT8
  derivative to this artifact.

## Limitations

- Quantization can reduce quality relative to the BF16 base model.
- FP8 KV cache introduces numerical loss independently of weight
  quantization.
- Native context length does not guarantee that every serving configuration
  fits in available VRAM.
- Vision and MTP paths require separate memory and compatibility validation.
- This checkpoint inherits the capabilities, risks, license, and intended-use
  guidance of the original Qwen3.8-27B model.
