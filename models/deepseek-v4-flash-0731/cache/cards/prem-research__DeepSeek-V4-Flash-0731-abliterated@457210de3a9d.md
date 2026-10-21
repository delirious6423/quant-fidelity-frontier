---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - abliteration
  - uncensored
  - deepseek
  - moe
  - fp8
  - fp4
pipeline_tag: text-generation
datasets:
  - Bahushruth/abliteration-harmful-enriched
  - mlabonne/harmful_behaviors
  - mlabonne/harmless_alpaca
  - HuggingFaceH4/no_robots
---

# DeepSeek-V4-Flash-0731 Abliterated (ARA)

Uncensored version of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) using **ARA (Arbitrary-Rank Ablation)** — first-principles per-module optimization (no refusal directions, no strength caps). Each steerable module is optimized directly with LBFGS so that harmless behavior is preserved while harmful-prompt outputs are steered toward the harmless output cloud.

Steered modules: attention `o_b_proj` + shared-expert `down_proj` + top-8 most-routed MoE experts' `down_proj` slices per layer. Layers 17–36, k=6 neighbors.

## Format

**Native original repo format** — fp8 e4m3 block-128 dense weights, fp4 e2m1 block-32 packed experts, MTP (speculative-decoding) weights included. Only the ~20 shards containing surgery-touched tensors (layers 17–36) were rewritten; everything else is byte-identical to the original repo. Loads and serves exactly like `deepseek-ai/DeepSeek-V4-Flash-0731`, no patches needed. `edited_experts.json` lists which expert slices were modified.

## Serving

vLLM (verified v0.26.0):
```bash
vllm serve <this-repo> --tensor-parallel-size 2 --kv-cache-dtype fp8 --max-model-len 8192
```

sglang (verified v0.5.16 — the MXFP4 MoE backend is required, the default triton path crashes on this arch):
```bash
python -m sglang.launch_server --model-path <this-repo> --tp 2 \
  --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4
```

No chat template ships with the tokenizer; use the bundled `encoding_dsv4.py` (raw format: `<｜User｜>...<｜Assistant｜></think>`).

## Metrics

| Metric | Original | Abliterated |
| :--- | :--- | :--- |
| Refusal rate (800 held-out harmful prompts) | 69% | **3%** |
| KL divergence on 100 harmless prompts | — | **0.044** |

Best Optuna trial params: `start_layer_index=17, end_layer_index=36, preserve_good_behavior_weight=0.876, steer_bad_behavior_weight=1.45e-4, overcorrect_relative_weight=0.472, neighbor_count=6`.

## Method

Per-module LBFGS optimization (fp32, strong-Wolfe line search) with row-norm-preserving reparameterization: preserve harmless outputs (MSE), steer harmful outputs toward the harmless kNN cloud, overcorrect away from original harmful outputs. Optuna TPE search (60 trials) over layer range and loss weights, objective = (refusal rate, KL) with KL budget 0.05. Surgery performed in bf16 on dequantized weights; edited tensors requantized back to the native fp8/fp4 formats (round-trip validated value-exact for untouched tensors).

Capture: 2000 harmful (enriched + mlabonne) / 2000 harmless (alpaca + no_robots) prompts. Eval: 800 held-out harmful, KL on 100 harmless.

## Datasets

| Dataset | Role |
| :--- | :--- |
| [Bahushruth/abliteration-harmful-enriched](https://huggingface.co/datasets/Bahushruth/abliteration-harmful-enriched) | Harmful prompts (enriched, multilingual, 33 categories) |
| [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) | Harmful prompts |
| [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) | Harmless prompts |
| [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | Harmless prompts |

## Disclaimer

This model has had safety guardrails removed and will comply with requests the original model would refuse. Released for research into AI alignment and safety mechanisms. The creator assumes no responsibility for downstream use.

## Acknowledgments

- [mlabonne](https://huggingface.co/mlabonne) — abliteration technique, harmful/harmless datasets, [original notebook](https://huggingface.co/blog/mlabonne/abliteration)
- [deepseek-ai](https://huggingface.co/deepseek-ai) — base model
- Created by [Bahushruth](https://huggingface.co/Bahushruth)
