---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-AWQ-UINT4/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - quark
  - amd-quark
  - quantization
  - uint4
  - awq
  - w4a16
  - moe
  - qwen
---

# Qwen3.8-Flash-Next-Quark-AWQ-UINT4

## Model Overview

- **Model Architecture:** Qwen4Exp (Qwen3.8-Flash-Next), hybrid linear-attention / sparse-attention MoE with 512 routed experts (10 active/token) and a shared expert
  - **Input:** Text
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI3xx (gfx942/gfx950)
- **ROCm:** 7.2.4
- **Operating System(s):** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html)
  - **Quantized layers:** `model.language_model.layers.*.mlp.experts.*.{gate,up,down}_proj` and the per-layer shared expert (**73,872** modules: 73,728 routed + 144 shared). All other layers stay BF16 (attention, linear attention, PLE n-gram table, vision tower, embeddings, `lm_head`, routers, MTP block)
  - **Weight quantization:** UINT4, group size 128, asymmetric (zero-point), Static
  - **Activation quantization:** None (weight-only, W4A16)
- **Calibration Dataset:** [`wikitext`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) (`wikitext_for_gptq_benchmark`), 512 samples, sequence length 512

UINT4 weight-only (W4A16) quantized version of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), produced with the **AWQ** algorithm using [AMD Quark](https://github.com/amd/Quark).

## Serving

Requires vLLM >= 0.31.0, the first release containing native Quark W4A16 INT4/UINT4 MoE support (`QuarkW4A16Int4MoEMethod`, [PR 48606](https://github.com/vllm-project/vllm/pull/48606)).

Serve with `vllm serve <repo-or-local-path>`. Verified on vLLM v0.33.0 (`vllm/vllm-openai-rocm:v0.33.0`), 1x MI355X, TP=1:

```bash
vllm serve amd/Qwen3.8-Flash-Next-Quark-AWQ-UINT4 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

The BF16 PLE n-gram table is CPU-offloaded, so the ~181 GB checkpoint fits one card (about 70 GiB on GPU).

To enable MTP speculative decoding, add `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.

## Evaluation

Single pass, TP=1, lm-evaluation-harness 0.4.13, vLLM 0.28.1rc1 + PR 48606 (since merged, in vLLM >= 0.31.0), AMD MI355X. Prompts use the model's own chat template via `--apply_chat_template`; sampling follows the official Qwen3.8-Flash-Next model card.

| Benchmark | Mode | n | bf16 base | AWQ | Recovery |
|---|---|---:|---:|---:|---:|
| GPQA-Diamond | thinking, `reasoning_effort=medium`, 40k | 198 | 84.34% | **85.35% +-2.52** | **101.2%** |
| GSM8K (strict-match) | thinking | 1319 | 97.42% | **97.65% +-0.42** | **100.2%** |
| GSM8K (strict-match) | non-thinking | 1319 | 95.98% | **95.38% +-0.58** | **99.4%** |

Recovery is the quantized score as a percentage of the bf16 base measured on the
same harness, same prompts and same sampling. The bf16 base was run at TP=4
(it does not fit one card); tensor-parallel degree changes sharding, not results.

Sampling settings (`max_length` 40960 throughout):

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
|---|---:|---:|---:|---:|---:|
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

## Checkpoint Reproduction

```bash
python3 examples/torch/language_modeling/llm_ptq/quantize_quark.py \
  --model_dir Qwen/Qwen3.8-Flash-Next \
  --output_dir <out> \
  --quant_scheme uint4_wo_128 --quant_algo awq \
  --dataset wikitext_for_gptq_benchmark \
  --data_type bfloat16 --device cuda --multi_gpu auto \
  --model_export hf_format --custom_mode quark \
  --export_weight_format real_quantized --skip_evaluation
```

Run from a checkout of [amd/Quark](https://github.com/amd/Quark). Calibration uses the argparse defaults: 512 samples x 512 tokens, batch size 1.

## License

This model is a derivative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and is distributed under the same license as the source model, the [Qwen Community License 1.0](LICENSE). See [NOTICE.txt](NOTICE.txt) for details.

Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.
