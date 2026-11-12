---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: vllm
language:
- multilingual
tags:
- ple-embedding
- per-token-table
- n-gram-embedding
- moe
- qwen4_exp
- vllm
- uint4
- awq
- amd-quark
- quantization
---

# Qwen3.8-Flash-Next-Quark-TokenPLE-AWQ-UINT4

## Model Overview

- **Model Architecture:** `Qwen4ExpForConditionalGeneration` (`model_type` `qwen4_exp`), a multimodal mixture-of-experts model with a PLE embedding at decoder layer index 1 (0-based; `ple_layer_ids: [2]` in `text_config`)
  - **Input:** Text, Image
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI3XX (gfx942/gfx950), AMD Ryzen AI RDNA 3+ (Strix Halo)
- **ROCm:** 7.2.3
- **PyTorch:** 2.13.0+git733fca1
- **Transformers:** 5.19.0
- **Operating System(s):** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/) 0.31.1rc1.dev88+g81198e97b with PLE per-token support [vLLM PR #64369](https://github.com/vllm-project/vllm/pull/64369)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html)
  - **Quantized layers:** routed experts `model.language_model.layers.*.mlp.experts.*.gate_proj`, `up_proj` and `down_proj` (73,728 modules = 48 layers x 512 experts x 3)
  - **Layers kept in BF16:** attention, router, shared expert, hyper-connection, embeddings, `lm_head`, the PLE table and the MTP layer (`mtp.*`); listed in `config.json::quantization_config.exclude` (922 entries)
  - **Weight quantization:** uint4, per-group (group size 128), asymmetric with int32 zero point, BF16 scales; AWQ (Quark `awq` algorithm, `up_proj` scales folded into `down_proj` per expert, see `config.json::quantization_config.algo_config`)
  - **Activation quantization:** None (weight-only)
- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Team, Alibaba Group), quantized with AMD Quark to AWQ uint4 g128
- **Modification:** the 102.4 GB BF16 PLE n-gram embedding table is replaced by a 1.27 GB per-token table `U`; all other weights are identical to the AWQ-quantized source checkpoint
- **Disclaimer:** this is not a release of the Qwen team, which has neither evaluated nor endorsed it

## PLE Table Replacement

- **Original table:** 320,001,536 rows x 160 values, BF16
- **Per-token table `U`:** shape [248,320, 2,560], BF16, 1,271,398,400 bytes; stored as `model.language_model.layers.1.ple.ple_embedding.per_token_table.weight` in `model-per_token_table.safetensors`
- **Construction:** `U` was obtained by calibration on the datasets listed in [`CORPUS_DATASHEET.md`](CORPUS_DATASHEET.md)
- **Mode selection:** `ple_embedding_mode: "per_token"` in `text_config` of `config.json`

## Model Quantization

- **Method:** quantized once with AMD Quark AWQ (real-quantized export) from the original BF16 model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with the original n-gram table
- **Expert weights:** those of [FILL IN: link to the original-table AWQ checkpoint]; not re-calibrated for `U`
- **Excluded layers:** no `--exclude_layers` option was passed; the 922 excluded entries are the script's model defaults and are recorded in `config.json`
- **Quantization command:**

```bash
python3 quantize_quark.py --model_dir <original Qwen3.8-Flash-Next> --output_dir <awq_dir> --quant_scheme uint4_wo_128 --quant_algo awq \
  --dataset token_file --calib_token_file <calib_128x512.npy> --num_calib_data 128 --seq_len 512 --batch_size 1 \
  --data_type bfloat16 --device cuda --multi_device --model_export hf_format --custom_mode quark --export_weight_format real_quantized --skip_evaluation
```

## Deployment

### Use with vLLM

- **Requirement:** vLLM with PLE per-token support [vLLM PR #64369](https://github.com/vllm-project/vllm/pull/64369) (open pull request)
- **Tested environment:** Docker image `vllm/vllm-openai-rocm:nightly` with the change from [vLLM PR #64369](https://github.com/vllm-project/vllm/pull/64369) applied to its bundled vLLM 0.31.1rc1.dev88+g81198e97b; vLLM was not reinstalled
- **Serve command:**

```bash
vllm serve <path-to-this-model> --tensor-parallel-size 1 --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --trust-remote-code --seed 1234 --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": true}'
```

- **PLE mode:** read from `config.json`; the server log shows the selected mode and `table_rows=248320`
- **Memory:** loading takes 70.54 GiB per GPU at tensor parallel 1, compared with 164.73 GiB for the AWQ checkpoint with the original table (`--engram-config '{"cpu_offload": false}'`)

## Evaluation

| Benchmark (thinking, effort xhigh) | Original n-gram table, mean ± SE (5 runs) | Per-token table U (this model), mean ± SE (5 runs) | Paired difference U − original, ± SE (per question) |
|---|---:|---:|---:|
| GSM8K (1,319 questions, 5-shot, strict-match) | 97.07 ± 0.11 | 96.38 ± 0.20 | -0.70 ± 0.23 |
| MMLU-Pro (12,032 questions, 5-shot) | 87.79 ± 0.07 | 86.72 ± 0.03 | -1.07 ± 0.13 |
| GPQA-Diamond (198 questions, 0-shot) | 92.12 ± 0.34 | 91.11 ± 0.71 | -1.01 ± 0.88 |
| BFCL V4 without web_search (overall, 20 categories) | 67.50 ± 0.34 | 65.55 ± 0.37 | -1.95 ± 0.51 |
| LiveCodeBench v6 (131 problems, 2025-02-01 to 2025-05-01), pass@1 | 87.63 ± 0.82 | 88.24 ± 0.92 | +0.61 ± 1.04 |

### Reproduction

- **Hardware:** AMD Instinct MI355X, tensor parallel 1
- **Environment:** Docker image `vllm/vllm-openai-rocm:nightly` with the change from [vLLM PR #64369](https://github.com/vllm-project/vllm/pull/64369) applied to its bundled vLLM; vLLM was not reinstalled
- **Harnesses:** lm-evaluation-harness 0.4.13 (task configurations at commit `82df34b` with local modifications); Berkeley Function Call Leaderboard (commit `dac44e7`); LiveCodeBench (runner `28fef95e`, dataset `livecodebench/code_generation_lite` revision `0fe84c39`)
- **Settings:** thinking enabled with reasoning effort xhigh, temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence penalty 0, `max_model_len` 262,144, maximum generation 253,952 tokens, `--gpu-memory-utilization` 0.85, seeds 1234 to 1238
- **Baseline:** the same AWQ uint4 g128 checkpoint with the original n-gram table, served with `--engram-config '{"cpu_offload": false}'` and the same settings

## License

- **Base license:** derivative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Team, Alibaba Group), released under the Qwen Community License 1.0 (Copyright (c) 2026 Qwen)
- **This repository:** distributed under the same license; the `LICENSE` file is the base model's, byte for byte (sha256 `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`)
- **NOTICE:** the base model repository ships no NOTICE file; no separate license is declared for the modifications
- **Modifications:** Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved. AMD modified `config.json`, `model.safetensors.index.json`, `model-00002-of-00003.safetensors` and `model-per_token_table.safetensors` (new) relative to the base release, and quantized the routed-expert weights (in `model-00002-of-00003.safetensors` and `model-00003-of-00003.safetensors`)
- **Re-written files:** `generation_config.json`, `preprocessor_config.json` and `processor_config.json` were re-written by the export tooling; the other files are unchanged
- **License conditions** (see the file): the copyright and permission notice in all copies; the model name displayed prominently in products with more than 100,000,000 monthly active users or US$ 20,000,000 monthly revenue; a separate license from Qwen for Model-as-a-Service or AI Work Assistant businesses
- **Calibration data:** the datasets used for calibration contain text under CC-BY-SA, ODC-By and CC-BY terms (sources listed in `CORPUS_DATASHEET.md`)

## Acknowledgements

- **Base model:** Qwen Team, Alibaba Group
- **Software:** vLLM (Apache-2.0), AMD Quark, lm-evaluation-harness, Berkeley Function Call Leaderboard, LiveCodeBench (MIT)
- **Evaluation data:** GPQA, MMLU-Pro, GSM8K, BFCL V4 and LiveCodeBench v6 were used for measurement only, never for calibration, each under its own terms
