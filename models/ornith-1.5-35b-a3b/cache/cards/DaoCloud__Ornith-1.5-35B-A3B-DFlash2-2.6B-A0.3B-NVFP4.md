---
license: apache-2.0
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
tags:
  - speculative-decoding
  - dflash2
  - draft-model
  - moe
  - speculators
  - vllm
  - nvfp4
---

# Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4

An NVFP4 Mixture-of-Experts DFlash2 drafter for [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) with a trained confidence head. The routed experts are quantized to NVFP4; every other module stays BF16.

## Results

| Setting | Value |
| --- | --- |
| Engine | vLLM |
| Hardware | NVIDIA H200, TP1 |
| Evaluation workers | 8 independent endpoints |
| Concurrency | 1 request per endpoint |
| Thinking mode | enabled |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Max new tokens | 4,096 |
| Context | native 262,144 |
| Draft tokens | 7 per verification step |

All results below use this NVFP4 drafter. Each benchmark is measured twice: once against the BF16 target [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), and once against the NVFP4 target [`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4).

| Benchmark | Samples | Target | p0 | p1 | p2 | p3 | p4 | p5 | p6 | AL |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AA-LCR | 100 | BF16 | 0.859 | 0.669 | 0.522 | 0.416 | 0.332 | 0.265 | 0.207 | 4.268 |
|  |  | NVFP4 | 0.865 | 0.678 | 0.526 | 0.421 | 0.335 | 0.269 | 0.208 | 4.301 |
| AIME26 | 30 | BF16 | 0.828 | 0.659 | 0.520 | 0.404 | 0.317 | 0.244 | 0.185 | 4.157 |
|  |  | NVFP4 | 0.819 | 0.652 | 0.514 | 0.402 | 0.312 | 0.242 | 0.184 | 4.126 |
| GSM8K | 1,319 | BF16 | 0.869 | 0.738 | 0.624 | 0.526 | 0.443 | 0.365 | 0.297 | 4.862 |
|  |  | NVFP4 | 0.870 | 0.741 | 0.624 | 0.525 | 0.443 | 0.364 | 0.298 | 4.865 |
| HumanEval | 164 | BF16 | 0.858 | 0.725 | 0.610 | 0.519 | 0.441 | 0.379 | 0.322 | 4.855 |
|  |  | NVFP4 | 0.855 | 0.719 | 0.605 | 0.511 | 0.435 | 0.372 | 0.316 | 4.812 |
| LiveCodeBench | 1,055 | BF16 | 0.790 | 0.598 | 0.450 | 0.340 | 0.258 | 0.197 | 0.150 | 3.782 |
|  |  | NVFP4 | 0.788 | 0.595 | 0.446 | 0.335 | 0.254 | 0.193 | 0.146 | 3.756 |
| LongBench-v2 | 402 | BF16 | 0.803 | 0.576 | 0.409 | 0.299 | 0.222 | 0.163 | 0.118 | 3.591 |
|  |  | NVFP4 | 0.796 | 0.567 | 0.401 | 0.292 | 0.216 | 0.158 | 0.114 | 3.543 |
| MATH500 | 500 | BF16 | 0.866 | 0.728 | 0.605 | 0.500 | 0.413 | 0.338 | 0.271 | 4.722 |
|  |  | NVFP4 | 0.866 | 0.726 | 0.602 | 0.497 | 0.411 | 0.336 | 0.270 | 4.707 |
| MBPP | 257 | BF16 | 0.819 | 0.653 | 0.517 | 0.411 | 0.328 | 0.262 | 0.208 | 4.199 |
|  |  | NVFP4 | 0.813 | 0.639 | 0.499 | 0.394 | 0.311 | 0.247 | 0.194 | 4.096 |
| MMSpec | 600 | BF16 | 0.726 | 0.501 | 0.345 | 0.242 | 0.174 | 0.124 | 0.087 | 3.198 |
|  |  | NVFP4 | 0.723 | 0.499 | 0.344 | 0.241 | 0.172 | 0.123 | 0.087 | 3.189 |
| MT-Bench | 80 | BF16 | 0.709 | 0.482 | 0.335 | 0.239 | 0.175 | 0.130 | 0.095 | 3.166 |
|  |  | NVFP4 | 0.699 | 0.471 | 0.327 | 0.235 | 0.171 | 0.126 | 0.091 | 3.120 |
| SWE-bench Pro | 731 | BF16 | 0.735 | 0.519 | 0.371 | 0.267 | 0.194 | 0.140 | 0.100 | 3.327 |
|  |  | NVFP4 | 0.731 | 0.515 | 0.367 | 0.264 | 0.190 | 0.137 | 0.097 | 3.300 |
| SpeedBench / coding | 80 | BF16 | 0.811 | 0.640 | 0.509 | 0.407 | 0.329 | 0.267 | 0.215 | 4.178 |
|  |  | NVFP4 | 0.810 | 0.633 | 0.496 | 0.393 | 0.315 | 0.255 | 0.205 | 4.108 |
| SpeedBench / humanities | 80 | BF16 | 0.701 | 0.474 | 0.318 | 0.214 | 0.146 | 0.100 | 0.066 | 3.019 |
|  |  | NVFP4 | 0.698 | 0.469 | 0.318 | 0.216 | 0.149 | 0.100 | 0.066 | 3.017 |
| SpeedBench / math | 80 | BF16 | 0.778 | 0.572 | 0.421 | 0.311 | 0.229 | 0.167 | 0.118 | 3.596 |
|  |  | NVFP4 | 0.758 | 0.559 | 0.409 | 0.303 | 0.224 | 0.165 | 0.118 | 3.537 |
| SpeedBench / multilingual | 80 | BF16 | 0.781 | 0.601 | 0.459 | 0.352 | 0.268 | 0.199 | 0.143 | 3.803 |
|  |  | NVFP4 | 0.773 | 0.591 | 0.454 | 0.347 | 0.265 | 0.199 | 0.148 | 3.778 |
| SpeedBench / qa | 80 | BF16 | 0.685 | 0.444 | 0.289 | 0.187 | 0.119 | 0.077 | 0.047 | 2.848 |
|  |  | NVFP4 | 0.688 | 0.456 | 0.298 | 0.197 | 0.129 | 0.086 | 0.054 | 2.909 |
| SpeedBench / rag | 80 | BF16 | 0.818 | 0.637 | 0.495 | 0.387 | 0.305 | 0.235 | 0.177 | 4.054 |
|  |  | NVFP4 | 0.816 | 0.637 | 0.496 | 0.388 | 0.302 | 0.229 | 0.174 | 4.042 |
| SpeedBench / reasoning | 80 | BF16 | 0.750 | 0.538 | 0.389 | 0.281 | 0.205 | 0.149 | 0.107 | 3.419 |
|  |  | NVFP4 | 0.753 | 0.545 | 0.393 | 0.286 | 0.208 | 0.151 | 0.108 | 3.444 |
| SpeedBench / roleplay | 80 | BF16 | 0.582 | 0.306 | 0.162 | 0.089 | 0.049 | 0.027 | 0.016 | 2.233 |
|  |  | NVFP4 | 0.571 | 0.292 | 0.149 | 0.082 | 0.046 | 0.025 | 0.014 | 2.180 |
| SpeedBench / stem | 80 | BF16 | 0.734 | 0.513 | 0.354 | 0.247 | 0.171 | 0.118 | 0.081 | 3.220 |
|  |  | NVFP4 | 0.732 | 0.511 | 0.356 | 0.248 | 0.174 | 0.123 | 0.085 | 3.229 |
| SpeedBench / summarization | 80 | BF16 | 0.778 | 0.571 | 0.408 | 0.296 | 0.206 | 0.136 | 0.080 | 3.476 |
|  |  | NVFP4 | 0.770 | 0.565 | 0.409 | 0.290 | 0.206 | 0.137 | 0.085 | 3.462 |
| SpeedBench / writing | 80 | BF16 | 0.638 | 0.379 | 0.233 | 0.151 | 0.103 | 0.070 | 0.047 | 2.621 |
|  |  | NVFP4 | 0.633 | 0.375 | 0.228 | 0.146 | 0.096 | 0.064 | 0.043 | 2.586 |

Accepted length is calculated from the raw serving counters, `p_i` is the fraction of draft rounds that survive through draft position `i`:

```text
p_i = accepted_per_pos[i] / draft_rounds
accepted_length = 1 + sum_i p_i
```

## Quantization

| Setting | Value |
| --- | --- |
| Tool | NVIDIA TensorRT Model Optimizer 0.46.0 |
| Algorithm | `W4A16_NVFP4`, weight-only |
| Block size | 16 |
| Calibration | none required (`algorithm: max`) |
| Quantized modules | routed experts only, `layers.*.mlp.experts.*.{gate,up,down}_proj` |
| Everything else | BF16 |
| Checkpoint size | 1,805,028,242 B, down from 5,277,111,386 B (-65.8%) |

## Architecture

| Setting | Value |
| --- | --- |
| Draft layers | 3 |
| Auxiliary hidden-state IDs | `[3, 12, 21, 30, 39]` |
| Hidden size | 2,048 |
| MoE FFN | 256 routed experts, top-8 routing |
| Routed expert intermediate | 512 |
| Shared expert | 1 expert, intermediate size 512 |
| Router | softmax, top-k normalization |
| Shared-expert gate | sigmoid |
| Attention | 32 Q heads / 4 KV heads |
| Head dimension | 128 |
| Attention window | sliding window 2,048 |
| Draft attention | non-causal inside the draft block |
| RoPE theta | 1e7 |
| Norm / activation | RMSNorm eps 1e-6 / SiLU |
| Dynamic convolution | kernel 2, group 16 |
| Candidate selector | rank 256, top-k 16 |
| Confidence head | included |
| Block | 7 draft tokens, `sample_from_anchor=true` |
| Draft vocabulary | 248,320 |
| Parameters | 2,638,552,065 total / 298,130,433 activated per token |
| Precision | NVFP4 routed experts, BF16 elsewhere |

## Example: Run with vLLM

This is a drafter checkpoint, not a standalone language model. Pair it with
Ornith-1.5-35B-A3B or a compatible quantized version:

```bash
vllm serve ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{
    "method": "dflash",
    "model": "DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4",
    "num_speculative_tokens": 7
  }'
```

The checkpoint currently requires a vLLM build containing:

- [`vllm-project/vllm#54154`](https://github.com/vllm-project/vllm/pull/54154) — `sample_from_anchor=true` query layouts for DFlash draft models.
- [`vllm-project/vllm#54475`](https://github.com/vllm-project/vllm/pull/54475) — MoE DFlash2 draft models.
- [`vllm-project/vllm#55668`](https://github.com/vllm-project/vllm/pull/55668) — optional DFlash2 confidence-head loading.

## Checkpoint

The checkpoint contains 6,971 tensors and 2,638,552,065 parameters.

| File | SHA256 |
| --- | --- |
| `model.safetensors` | `2f87b58b3f7491faf39a84b3074a2ff63e93d3f7c053cf6d59916f3088a1dfc7` |

## Acknowledgements

This work builds on [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), `vllm-project/speculators`, `vllm-project/vllm`, and `NVIDIA/TensorRT-Model-Optimizer`.

## License

Apache-2.0.
