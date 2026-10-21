---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: sglang
language: [en, ar, zh, fr, de, ja, ko, es, pt, it, vi, th, id, hi, ru, pl]
tags: [nvfp4, fp4, modelopt, quantized, w4a16, fp8-kv, moe, liquid, lfm2.5, blackwell, sglang, reasoning, tool-use]
thumbnail: https://huggingface.co/GestaltLabs/LFM2.5-24B-A2B-NVFP4-W4A16/resolve/main/assets/banner.png
---

<p align="center"><img src="assets/banner.png" alt="LFM2.5-24B-A2B NVFP4 W4A16" width="100%"></p>

<p align="center">
<img alt="format" src="https://img.shields.io/badge/format-NVFP4%20W4A16-E69F00">
<img alt="kv" src="https://img.shields.io/badge/KV%20cache-FP8%20e4m3-0072B2">
<img alt="toolkit" src="https://img.shields.io/badge/NVIDIA%20ModelOpt-0.47.0-76B900">
<img alt="engine" src="https://img.shields.io/badge/SGLang-0.5.22%20tested-4B5563">
<img alt="gpu" src="https://img.shields.io/badge/GPU-Blackwell%20sm__120-76B900">
</p>

**Support on [Ko-fi](https://ko-fi.com/djlougen) or X Money: [@DJLougen](https://x.com/DJLougen)**

# LFM2.5-24B-A2B · NVFP4 W4A16

## Routed experts 4.5 bpw (NVFP4) · everything else BF16 · 14.95 GB download (4.99 bpw overall) · 15.9 GiB resident · 98.6% of BF16 accuracy on 4 multiple-choice suites

A 4-bit build of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), Liquid AI's 24B-parameter, 2B-active
reasoning and agent model. The 64 routed experts per MoE layer (95.7% of all parameters) are stored in NVFP4 and compute in 16-bit;
attention, short-convolution blocks, the two dense layers, routers and embeddings are left exactly as Liquid released them. It is built with
**NVIDIA Model Optimizer** and ships in ModelOpt's unified checkpoint format, which SGLang loads natively.

Official source: this repository. If you'd like to share the model, please link here rather than re-uploading.

<p align="center"><img src="assets/accuracy.png" width="92%"></p>

## At a glance

| | BF16 original | **this repo** |
|---|---|---|
| download | 47.94 GB | **14.95 GB** |
| weights resident (SGLang) | 45.24 GiB | **15.91 GiB** |
| KV-cache pool, same settings | 832,077 tokens | **3,283,030 tokens (3.9×)** |
| accuracy retained, MMLU · HellaSwag · ARC-C · Winogrande | 100% | **98.61%** |
| next-token agreement with BF16 (WikiText / C4) | 100% | **86.68% / 83.12%** |

<p align="center"><img src="assets/memory.png" width="100%"></p>

## Quick start (SGLang)

```bash
pip install "sglang[all]==0.5.22"
python -m sglang.launch_server \
  --model-path GestaltLabs/LFM2.5-24B-A2B-NVFP4-W4A16 \
  --moe-runner-backend marlin \
  --reasoning-parser qwen3-thinking --tool-call-parser lfm2 \
  --context-length 131072 --mem-fraction-static 0.80
```

`--moe-runner-backend marlin` is **required**: SGLang's default (`auto`) picks a W4A4 kernel for NVFP4 MoE layers and fails on this
weight-only checkpoint with `AttributeError: 'FusedMoE' object has no attribute 'g1_scale_c'`. The FP8 KV cache is picked up from
`hf_quant_config.json` automatically. The model always thinks before answering, so give it a generous `max_tokens`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
          "required": ["city"]}}}]
r = client.chat.completions.create(model="default", tools=tools, max_tokens=4096,
        messages=[{"role": "user", "content": "What's the weather in Toronto in celsius?"}])
print(r.choices[0].message.reasoning_content)   # the model's thinking
print(r.choices[0].message.tool_calls)          # [get_weather(city="Toronto", unit="c")]
```

Liquid's recommended sampling (`temperature 0.1`, `top_k 50`, `repetition_penalty 1.05`) is in `generation_config.json` and SGLang applies it by default.

## How close is it to BF16?

<p align="center"><img src="assets/fidelity.png" width="100%"></p>

Every comparison is **paired**: the same items, the same prompts, the same harness version and the same batch size for BF16 and for this
build, so per-item differences can be counted. MMLU drops by about one point and that drop is statistically real; HellaSwag, ARC-Challenge
and Winogrande move by less than their confidence intervals. TruthfulQA-MC2 rises, which should not be read as an improvement: that metric
also goes up when a model's confidence is merely flattened, and it returned to the BF16 level when the KV cache was kept in BF16 (ablation below).

<details>
<summary><b>Exact numbers</b></summary>

**Accuracy** (lm-evaluation-harness 0.4.13, loglikelihood, seed 1234; Δ = this repo − BF16 in points; paired bootstrap 95% CI; McNemar exact test on items that changed)

| task | n | BF16 % | this repo % | Δ pts (95% CI) | BF16-only ✓ / quant-only ✓ | McNemar p |
|---|---|---|---|---|---|---|
| MMLU (5-shot) | 14,042 | 62.87 | 61.84 | -1.03 (-1.54 to -0.53) | 738 / 594 | 8.79e-05 |
| HellaSwag (10-shot) | 10,042 | 69.84 | 70.02 | +0.18 (-0.24 to +0.63) | 236 / 254 | 0.443 |
| ARC-Challenge (25-shot) | 1,172 | 61.52 | 60.84 | -0.68 (-2.13 to +0.77) | 42 / 34 | 0.422 |
| Winogrande (5-shot) | 1,267 | 64.48 | 62.51 | -1.97 (-4.42 to +0.24) | 126 / 101 | 0.111 |
| TruthfulQA-MC2 (0-shot) | 817 | 51.16 | 53.74 | +2.58 (+1.34 to +3.84) | – | – (continuous) |

WikiText-2 rolling word perplexity, 4,096-token window: BF16 18.932, this repo 18.428 (-2.66%).
Raw-text perplexity is a secondary metric for an instruction-tuned reasoning model: across the candidates below it disagreed in sign with
token-level divergence, so it was not used to choose the recipe.

**Token-level divergence** (teacher-forced, 48 × 4,096-token windows per set, every window starts with BOS; reference = BF16 top-128 log-probs + tail bucket)

| set | tokens | KL mean (95% CI) | KL p50 | KL p99 | top-1 agreement | top-5 overlap |
|---|---|---|---|---|---|---|
| WikiText-2 test | 196,560 | 0.0849 (0.0793–0.0911) | 0.0412 | 0.739 | 86.68% | 83.82% |
| C4 validation | 196,560 | 0.1339 (0.1255–0.1424) | 0.0575 | 1.389 | 83.12% | 82.18% |

Control: BF16 scored against its own cached reference gives KL = 0 at every percentile and 100% agreement, so the harness adds no divergence of its own.
</details>

## What's quantized

| part | params | share | bytes on disk | bpw |
|---|---|---|---|---|
| routed experts (NVFP4 W4A16) | 22.951 B | 95.74% | 12.910 GB | 4.50 |
| short-conv blocks (BF16) | 0.504 B | 2.10% | 1.007 GB | 16.00 |
| embeddings, tied LM head (BF16) | 0.262 B | 1.09% | 0.524 GB | 16.00 |
| dense FFN, layers 0-1 (BF16) | 0.145 B | 0.60% | 0.289 GB | 16.00 |
| attention (BF16) | 0.105 B | 0.44% | 0.210 GB | 16.00 |
| routers + expert bias | 0.005 B | 0.02% | 0.010 GB | 16.01 |
| norms | 0.000 B | 0.00% | 0.000 GB | 16.00 |
| **total** | **23.972 B** | 100% | **14.951 GB** | **4.99** |

NVFP4 = 4-bit E2M1 values with one FP8 (E4M3) scale per 16 weights plus one FP32 scale per tensor, so 4.50 bits per expert weight
including scales. The gate (w1) and up (w3) projections of each expert share one global scale, which SGLang's Marlin path requires (verified
for all 2,432 pairs). Counts are computed from the tensor headers of the files in this repo.

## How it was made

* **Toolkit:** NVIDIA Model Optimizer 0.47.0, recipe system at the matching `0.47.0` source tag.
* **Scheme:** NVFP4 weights on the routed experts only (W4A16, activations stay BF16), static per-block scales chosen by an **MSE search over all
  FP8 scale candidates** (`method: mse`, `fp8_scale_sweep: true`), FP8 KV-cache cast. Everything else is byte-identical to the release.
* **Calibration:** 512 chat samples (580,554 tokens) from `HuggingFaceH4/ultrachat_200k`, seed 1234. The MSE weight scales depend only on the weights, not on these samples.
* **Reproducible:** re-running the recipe on a second machine reproduced all 22,202 tensors bit for bit.
* **Notice of changes (LFM Open License v1.0, §4(b)):** the routed-expert weights were quantized to NVFP4 and the config files carry a quantization configuration. No other weights were changed.

<details>
<summary><b>ModelOpt recipe</b></summary>

```yaml
# Ablation of nvfp4_experts_only_mse-kv_fp8_cast for LFM2.5-24B-A2B: identical except the expert INPUT
# (activation) quantizers are removed -> NVFP4 W4A16 experts with MSE (FP8-scale-sweep) static weight scales.
# Purpose: isolate the cost of NVFP4 activation quantization (W4A4 vs W4A16) with everything else held fixed.
# Stock nvfp4_mlp_weight_only.yaml matched 0 quantizers on LFM2.5 and uses block size 32, so it is not used.

imports:
  base_disable_all: configs/ptq/units/base_disable_all
  default_disabled_quantizers: configs/ptq/units/default_disabled_quantizers
  nvfp4_static: configs/numerics/nvfp4_static
  kv_fp8_cast: configs/ptq/units/kv_fp8_cast

metadata:
  recipe_type: ptq
  description: >-
    Static NVFP4 weight scales from MSE FP8-scale sweep on expert weights only (W4A16; activations BF16),
    plus FP8 KV-cache cast mode. Ablation of nvfp4_experts_only_mse-kv_fp8_cast.
quantize:
  algorithm:
    method: mse
    fp8_scale_sweep: true
    layerwise:
      enable: false
  quant_cfg:
    - $import: base_disable_all
    - quantizer_name: '*.experts.*weight_quantizer'
      cfg:
        $import: nvfp4_static
    - $import: kv_fp8_cast
    - $import: default_disabled_quantizers
```
</details>

## How the recipe was chosen

<p align="center"><img src="assets/recipe_search.png" width="95%"></p>

Six builds went through the same paired suite. The three W4A4 builds, which also quantize the expert **activations** to FP4, lost 1.8 to 2.1
MMLU points against 1.0 for this build. Among the W4A4 builds, MSE scale selection gave the lowest divergence; their MMLU drops overlap
within the confidence intervals, and the Local-Hessian build also changes attention and convolution to FP8, so it is not a pure test of the
scale method. The plain round-to-nearest build that quantizes every linear layer kept MMLU (+0.3) but had the highest divergence of all
candidates and a significant HellaSwag drop. Keeping the activations in 16-bit gave the lowest divergence of the shipped-format candidates;
the final ablation (BF16 KV cache) shows how much of what remains comes from the FP8 KV cache.

<details>
<summary><b>Candidate table</b></summary>

| build | KL WikiText | KL C4 | MMLU Δ pts |
|---|---|---|---|
| MSE · W4A16 experts · BF16 KV  (ablation) | 0.073 | 0.117 | pending |
| MSE · W4A16 experts · FP8 KV **(this repo)** | 0.085 | 0.134 | -1.03 |
| MSE · W4A4 experts · FP8 KV | 0.126 | 0.188 | -2.12 |
| max · W4A4 experts · FP8 KV | 0.136 | 0.206 | -1.77 |
| Local-Hessian · W4A4 experts+MLP · FP8 attn/conv | 0.149 | 0.216 | -1.98 |
| RTN · W4A16 · all linears (llm-compressor) | 0.159 | 0.253 | +0.26 |
</details>

## Limitations

* **Not measured yet:** think-mode generation tasks (GSM8K, MATH, IFEval), tool-calling accuracy (BFCL), multilingual quality, context beyond
  4,096 tokens, and throughput. The tool-calling and reasoning paths were smoke-tested on SGLang (a correct tool call and a correct boxed answer),
  which shows they work, not how well.
* **Where the accuracy numbers come from:** the paired suite ran in Hugging Face transformers with ModelOpt's simulated quantization of these exact
  weights (BF16 arithmetic). Served through SGLang's Marlin kernels the arithmetic order differs slightly.
* **Engines:** tested on SGLang 0.5.22 only, on an RTX PRO 6000 Blackwell (sm_120). vLLM and other GPUs are untested. Repeated identical
  temperature-0 requests were not bit-identical in our tests (3 to 4 distinct outputs out of 4), including with Marlin's atomic-add reduce
  disabled; the cause was not isolated. `--enable-deterministic-inference` stalled the BF16 reference, so it is not used here.
* **MMLU** loses about one point (statistically significant). Liquid describes the base model as not intended for knowledge-heavy tasks on its own.

## License

[LFM Open License v1.0](LICENSE), the base model's licence, a copy of which is included. This is a modified version of LiquidAI/LFM2.5-24B-A2B
(see *Notice of changes* above); the model and its copyright belong to Liquid AI, Inc. Commercial use is licensed only for legal entities below
US$10 million in annual revenue (see the licence for the exact terms).

## Files

`model-0000*-of-00002.safetensors` (weights, ModelOpt unified format), `hf_quant_config.json`, `config.json`, tokenizer and chat template,
`SHA256SUMS`, and `results/` with the data behind every number and chart on this page (`card_data.json`, per-task results, the recipe, the evaluation script and the serving log lines).

**Support on [Ko-fi](https://ko-fi.com/djlougen) or X Money: [@DJLougen](https://x.com/DJLougen)**
