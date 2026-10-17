---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - abliterated
  - qwen
  - qwen3
  - qwen3.8
  - uncensored
  - ai-red-team
  - red-teaming
  - nvfp4
  - fp4
  - fp8
  - mixed-precision
  - compressed-tensors
  - vllm
  - vision-language
  - function-calling
  - reasoning
  - mtp
---

<div align="center">

<a href="https://www.orcarouter.ai" target="_blank">
  <img src="https://www.orcarouter.ai/orca-logo-classic.png" alt="OrcaRouter" width="110">
</a>

<h1>Qwen3.8-27B-Uncensored-NVFP4</h1>

<p><em>An abliterated (refusal-removed) &amp; dynamic mixed-precision <strong>NVFP4 + FP8</strong> build of Qwen's Qwen3.8-27B — for Blackwell FP4</em></p>

<p>
<a href="https://www.orcarouter.ai/models/obsidian/qwen3.8-27b"><img src="https://img.shields.io/badge/%E2%96%B6%20Run%20via-OrcaRouter%20API-FF6A00" alt="Run via API"></a>
<a href="https://api.orcarouter.ai/v1"><img src="https://img.shields.io/badge/API-api.orcarouter.ai%2Fv1-1E6FEB" alt="API endpoint"></a>
</p>

<p>
<a href="https://www.orcarouter.ai"><img src="https://img.shields.io/badge/Website-orcarouter.ai-1E6FEB" alt="Website"></a>
<a href="https://www.orcarouter.ai/models"><img src="https://img.shields.io/badge/OrcaRouter-Model%20Catalog-2EA043" alt="Model Catalog"></a>
<a href="https://www.orcarouter.ai/models/obsidian/qwen3.8-27b"><img src="https://img.shields.io/badge/OrcaRouter-Model%20Card-6F42C1" alt="Model Card"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-4C8BF5" alt="License"></a>
<img src="https://img.shields.io/badge/Precision-NVFP4%20%2B%20FP8%20Mixed-00A67E" alt="NVFP4 + FP8">
<img src="https://img.shields.io/badge/Context-262K-FF8800" alt="262K context">
<img src="https://img.shields.io/badge/Vision-Language-9B59B6" alt="Vision-Language">
<img src="https://img.shields.io/badge/Speculative-MTP-E36209" alt="MTP">
</p>

<p><strong>One Gateway. Every Model.</strong> — Route Smarter · Ship Safer · Spend Less.</p>

<p>
<a href="https://www.orcarouter.ai/models/obsidian/qwen3.8-27b"><strong>Run via API</strong></a> ·
<a href="https://api.orcarouter.ai/v1">API Endpoint</a> ·
<a href="https://www.orcarouter.ai">Website</a> ·
<a href="https://www.orcarouter.ai/models">Model Catalog</a> ·
<a href="https://www.orcarouter.ai/models/obsidian/qwen3.8-27b">Model Card</a> ·
<a href="https://github.com/Continuum-AI-Corp">GitHub</a> ·
<a href="https://discord.gg/yAh6Tex6kx">Discord</a> ·
<a href="https://x.com/OrcaRouter">X</a>

</p>

</div>

---

> An **abliterated** (refusal-removed) and **dynamic mixed-precision NVFP4 + FP8 quantized**
> build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — a 27B-parameter dense,
> hybrid-attention (Gated DeltaNet linear + full attention) native vision-language model with
> flexible thinking control, tool-calling, and an MTP speculative-decoding head. This build removes
> the safety refusal direction, then quantizes the bulk feed-forward layers to **4-bit NVFP4** while
> keeping precision-sensitive layers and the KV cache at **FP8**, so accuracy is better preserved
> than uniform W4A4. **262K context, tools + reasoning + MTP + vision preserved.**
> Browse all models in the [OrcaRouter Model Catalog](https://www.orcarouter.ai/models).
>
> ### ▶ This model is deployed as a hosted API
> **Run it instantly on [OrcaRouter](https://www.orcarouter.ai/models/obsidian/qwen3.8-27b)** — OpenAI-compatible, no setup, 262K context with tools + reasoning. Endpoint: **[`api.orcarouter.ai/v1`](https://api.orcarouter.ai/v1)** · model `obsidian/qwen3.8-27b`. Grab a key at [orcarouter.ai](https://www.orcarouter.ai) (`sk-orca-...`).
>
> **Sibling releases:** &nbsp;•&nbsp; [`Qwen3.8-27B-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) — BF16 source &nbsp;•&nbsp; [`Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) — block-FP8 for vLLM &nbsp;•&nbsp; [`Qwen3.8-27B-Uncensored-GGUF`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) — 2-bit→16-bit for llama.cpp &nbsp;•&nbsp; [`Qwen3.8-27B-Uncensored-MLX`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX) — MLX for Apple Silicon (2 / 4 / 8-bit).

---

## ⚠️ Disclaimer — read before use

This model has had its **safety alignment substantially removed** via *abliteration*
(orthogonalizing the refusal direction out of the residual stream). As a direct consequence:

- **It will comply with harmful, unethical, offensive, or illegal requests** that the
  original `Qwen3.8-27B` would refuse. It has no meaningful built-in guardrails.
- It is released **strictly for legitimate research** — interpretability, AI-safety and
  refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
- **You assume full responsibility and liability** for how you use it and for everything it
  generates. Do not deploy it to end users or in production without adding your own safety,
  moderation, and abuse-prevention layers.
- Use must comply with the **[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)**
  inherited from the base model, and all laws and regulations that apply to you.
- The authors and uploaders **accept no liability** for any misuse or harm arising from this
  model. Its outputs do **not** reflect the views of the uploaders or of Qwen / Alibaba.

By downloading or using this model you acknowledge and accept the above.

---

## Model details

| | |
|---|---|
| **Base model** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Architecture** | `Qwen3_5ForConditionalGeneration` — 64 layers, hidden 5120, hybrid **Gated DeltaNet** (48 linear-attention + 16 full-attention, interval 4), native VL tower + MTP head |
| **Modification** | Abliteration (refusal-direction removal) **then** dynamic mixed-precision NVFP4 + FP8 quantization |
| **Quantization** | **Mixed-precision** (compressed-tensors) — NVFP4 (W4A4) on the FFN of layers 0–55, FP8 (W8A8 dynamic) on attention / GDN projections / the last 8 layers' FFN / `lm_head`, and a static **FP8 KV cache** |
| **Format** | safetensors, resharded to **≤ 5 GB shards** (5 + 1 shards, **23.4 GB**, 1968 tensors) |
| **Precision** | FP4 E2M1 group-16 (+ FP8-E4M3 block scale + FP32 global scale) and FP8 E4M3; vision tower / norms / GDN `in_proj_a/b` / embeddings / **MTP head** kept in BF16 |
| **Preserved** | Full **vision-language tower** and **MTP speculative-decoding head** (drop-in for the base) |
| **Context** | 262,144 tokens |

### Abliteration

Refusal-direction removal following **Arditi et al. (2024)**, *Refusal in Language Models Is
Mediated by a Single Direction*. A single refusal direction `r` (k = 1) is estimated as the
massive-activation–masked mean-difference of harmful − harmless last-token residuals at
**layer 38** (`round(0.6 × 64)`), on AdvBench (harmful) vs Alpaca (harmless). `r` is then
orthogonalized out of every **residual-writing** matrix — `W' = W − r(rᵀW)` — computed in
float32:

| Component | matrices edited |
|---|---|
| `self_attn.o_proj` (16 full-attention layers + MTP) | 17 |
| `linear_attn.out_proj` (48 linear-attention / GDN layers) | 48 |
| `mlp.down_proj` (64 layers + MTP) | 65 |
| `embed_tokens` (row space) | 1 |
| **Total** | **131** |

The **vision tower is untouched** and the **MTP head is abliterated consistently** with the
main model, so speculative decoding keeps working. This is the same abliterated BF16 base as the
[`FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8),
[`GGUF`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) and
[`BF16`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) releases — only the
quantization differs.

### Dynamic mixed-precision (NVFP4 + FP8) scheme

Rather than quantizing every linear uniformly, precision-sensitive layers are kept at FP8 while
only the bulk feed-forward layers go to 4-bit:

| Component | Precision | # linears |
|---|---|---:|
| MLP `gate/up/down`, layers **0–55** | **NVFP4** — FP4 E2M1, group 16, FP8-E4M3 block scale + FP32 global scale (W4A4) | 168 |
| `self_attn.{q,k,v,o}_proj`, GDN `in_proj_qkv/in_proj_z/out_proj` | **FP8** — W8A8, per-channel weight, dynamic per-token activation | 208 |
| MLP `gate/up/down` of the **last 8 layers (56–63)**, `lm_head` | **FP8** (W8A8 dynamic) | 25 |
| **KV cache** | **FP8** — static, per-tensor | — |
| Vision tower, GDN `in_proj_a/in_proj_b`, all norms, embeddings, **MTP head** | **BF16** (unquantized) | 206 |

- **Weights:** round-to-nearest. NVFP4 packs FP4 (E2M1) in groups of 16 with an FP8-E4M3 block
  scale and an FP32 per-tensor global scale; FP8 uses per-output-channel scales.
- **Activations:** NVFP4 layers use dynamic per-token FP4 with a calibrated global scale; FP8
  layers use dynamic per-token FP8 (no static activation scale).
- **KV cache:** static per-tensor FP8, calibrated.
- **Calibration:** 512 samples — 75% [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)
  + 25% compliant harmful completions, `enable_thinking=False`, sequence length 2048 — used only
  for the NVFP4 activation global scales and the static FP8 KV-cache scales, keeping calibration on
  the activation distribution the abliterated model actually produces.
- Built with **`llmcompressor`** (`QuantizationModifier`, two config groups + `kv_cache_scheme`),
  `format: mixed-precision`. Split: **168** NVFP4 / **233** FP8 / **206** BF16 linears.

vLLM serves this through the **`compressed-tensors`** path: the FP4 layers use FP4 tensor cores on
**Blackwell**, while the FP8 layers run on Hopper-class and newer.

## Intended use

- Research into refusal mechanisms, alignment, and interpretability.
- Red-teaming and safety / robustness evaluation in controlled environments.
- Uncensored generation for authorized, lawful research settings.

### Out of scope

- Any use that violates the base model's Apache 2.0 license or applicable law.
- Deployment to the public or to end users without additional safety and moderation layers.
- Generating content intended to harm, harass, defraud, or endanger people.

## Evaluation

Abliteration is a **weight edit shared across all releases** of this model, so the refusal
behavior of this checkpoint tracks the BF16 / FP8 builds. On the byte-identical-scheme
[`FP8` build](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8), harmful-prompt
refusal collapses from **64–99%** (base) to **0–6%** (thinking off) and **≤ 1.7%** (thinking on),
while benign over-refusal *drops* (XSTest-safe 5.6% → 0.4%) and capability stays within ±1.3 pts
of the base (MMLU 84.3 → 84.7, MMLU-Pro 77.6 → 76.8, GSM8K 90.0 → 88.7, CMMLU 81.4 → 80.8). See
that model card for the full tables.

> **Quant-specific numbers pending.** Capability-retention and perplexity for *this*
> NVFP4 + FP8 mixed-precision checkpoint have **not yet been measured** — the FP4 layers require
> **Blackwell** FP4 tensor cores to run natively, and this build is released for evaluation on
> that hardware. Numbers will be added here once benchmarked. As a mixed 4-bit/8-bit checkpoint it
> is expected to trade a little accuracy for size versus the FP8 build; the dynamic split (only the
> less-sensitive FFN layers at FP4) is designed to keep that loss small.

### Multimodal (vision)

The **vision tower is preserved** — all 167 `visual.*` weight tensors are kept in BF16 and the
merger / image + video preprocessor configs are intact, so this stays a full vision-language model
(`Qwen3_5ForConditionalGeneration`), a drop-in for the base. Abliteration only edits the
*language-model* residual writers, so image understanding is architecturally unaffected (and
image-conditioned refusals are reduced along with text ones). Serve **without**
`--language-model-only` to use vision.

## Usage

### Self-host with vLLM (OpenAI-compatible)

Requires a recent vLLM (**≥ 0.27**, with `compressed-tensors`). The FP4 layers need a **Blackwell**
GPU (B200 / GB200 / RTX 50-series) for native FP4 tensor cores; the FP8 layers run on Hopper-class
and newer.

```bash
docker run -d --name qwen38-uncensored-nvfp4 --gpus all --ipc=host --shm-size=8g \
  -v /path/to/Qwen3.8-27B-Uncensored-NVFP4:/model:ro \
  -p 8000:8000 vllm/vllm-openai:v0.27.1 \
  --model /model --served-model-name Qwen3.8-27B-Uncensored \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 --max-num-seqs 96 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

> The mixed-precision `quantization_config` (including the FP8 `kv_cache_scheme`) is read from
> `config.json` — **do not** pass `--quantization` or `--kv-cache-dtype`. `--speculative-config mtp`
> enables the preserved MTP draft head.

### Reasoning (thinking) toggle

Thinking is **on by default** (Qwen3.8). Toggle it per request via `chat_template_kwargs`; the
reasoning trace is returned in the `reasoning` field (`--reasoning-parser qwen3`).

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen3.8-27B-Uncensored",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.reasoning)  # thinking trace
print(resp.choices[0].message.content)    # final answer
```

### Tool calling

Standard OpenAI `tools` + assistant `tool_calls` + `role: tool` result messages are supported,
including multi-turn (feed the tool result back for a follow-up answer). Parsed by
`--tool-call-parser qwen3_coder`.

### Via OrcaRouter (hosted API — no setup)

Served on **[OrcaRouter](https://www.orcarouter.ai/models)** through the OpenAI-compatible
gateway (262K context, tools + reasoning). Grab an API key at
[orcarouter.ai](https://www.orcarouter.ai) (`sk-orca-...`).

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.orcarouter.ai/v1", api_key="sk-orca-...")
resp = client.chat.completions.create(
    model="obsidian/qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

## Hardware requirements & performance

**Software**
- vLLM **≥ 0.27** with `compressed-tensors` (Qwen3.5/3.8 support) — e.g. `vllm/vllm-openai:v0.27.1`.

**Compute**
- The **NVFP4 (FP4) layers require a Blackwell GPU** (B200 / GB200 / RTX 50-series) for native FP4
  tensor cores. The FP8 layers run on Hopper (H100 / H200) and newer.

**Memory**
- Weights: **~23 GB** (mixed 4-bit / 8-bit), vs the ~56 GB BF16 checkpoint.
- Minimum **~32 GB VRAM** for weights + a small KV cache; the full 262K context needs substantial
  extra KV cache (the checkpoint already stores the KV cache in FP8).
- Recommended: a single Blackwell **B200** (or larger) for the full FP4 path.

**Throughput / concurrency**
- Continuous batching; concurrency bounded by `--max-num-seqs` and the KV cache that fits after
  weights are loaded. The MTP draft head gives a large decode speedup on real workloads.

## Bias, risks, and limitations

- **Safety guardrails removed** — the model will produce harmful, biased, or offensive content
  on request. See the disclaimer above.
- It inherits any biases and limitations of the base `Qwen3.8-27B`.
- Mixed NVFP4 + FP8 is **not lossless** versus BF16; quant-specific capability impact for this
  build is not yet measured (see Evaluation).
- The FP4 layers require Blackwell hardware to run natively; on pre-Blackwell GPUs the FP4 path is
  unavailable.

## License

**Apache 2.0**, inherited from the base model
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). Abliteration and quantization do
not change the underlying license obligations.
