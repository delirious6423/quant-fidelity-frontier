---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - orcarouter/Qwen3.8-Flash-Next-Uncensored
  - primitive-ai/Qwen3.8-Flash-Next-NVFP4
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - nvfp4
  - quantized
  - vllm
  - abliterated
  - uncensored
  - qwen
  - qwen3.8
  - flash-next
  - moe
  - vision-language
  - function-calling
  - reasoning
  - mtp
  - ai-red-team
  - red-teaming
  - single-gpu
---

<p align="center">
  <img src="https://img.shields.io/badge/size-~174%20GB-F98424?style=flat-square" alt="size ~174 GB">
  <img src="https://img.shields.io/badge/formats-NVFP4%20experts%20+%20BF16-F98424?style=flat-square" alt="formats NVFP4 experts + BF16">
  <img src="https://img.shields.io/badge/runtime-vLLM%20qwen38--flash--next-F98424?style=flat-square" alt="runtime vLLM">
  <img src="https://img.shields.io/badge/hardware-1x%20GB10%20/%2096GB%20Blackwell-3DD13D?style=flat-square" alt="hardware">
  <img src="https://img.shields.io/badge/license-Qwen%20Community%201.0-F98424?style=flat-square" alt="license">
</p>

<h1 align="center">Qwen3.8-Flash-Next-Uncensored-NVFP4</h1>

<p align="center">
Abliterated Flash-Next, packed so it serves on <b>one GB10 / 96&nbsp;GB Blackwell</b>
the same way <a href="https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4">Primitive's NVFP4</a> does.
</p>

NVFP4 graft of [`orcarouter/Qwen3.8-Flash-Next-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) onto the Primitive experts-only RTN recipe. Routed experts are NVFP4 (group 16); the 51B PLE n-gram table stays BF16 so `VLLM_PLE_CPU_OFFLOAD` works. This is **not** the 336 GB BF16 source and **not** a TP=8 serve.

---

## ⚠️ Disclaimer — read before use

This model has had its **safety alignment substantially removed** via *abliteration* (orthogonalizing the refusal direction out of the residual stream), inherited from the OrcaRouter BF16 parent. As a direct consequence:

- **It will comply with harmful, unethical, offensive, or illegal requests** that the original `Qwen3.8-Flash-Next` would refuse. It has no meaningful built-in guardrails.
- It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
- **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
- Use must comply with the **[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)** inherited from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), and all laws and regulations that apply to you. (OrcaRouter labeled the BF16 parent Apache 2.0; the Qwen source card is Community 1.0, which is what this checkpoint follows.)
- The authors and uploaders **accept no liability** for any misuse or harm. Its outputs do **not** reflect the views of the uploaders, OrcaRouter, Primitive, or Qwen / Alibaba.

By downloading or using this model you acknowledge and accept the above.

---

## What this is

| | |
|---|---|
| **Abliterated parent** | [`orcarouter/Qwen3.8-Flash-Next-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) (BF16, 336 GB, TP=8 in their vLLM example) |
| **Quant parent** | [`primitive-ai/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4) (experts-only RTN, BF16 PLE) |
| **Qwen source** | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **Architecture** | `Qwen4ExpForConditionalGeneration` (`qwen4_exp`) — 48 layers, hidden 2560, 512 fused experts top-10 + shared expert, hybrid attention, native vision + video, MTP head |
| **On disk** | ~174 GB safetensors |
| **Quant** | Routed `gate` / `up` / `down` NVFP4 group 16, `input_scale=1.0`, producer `modelopt 0.46.0-compatible-rtn`. Weights-only, no calibration. |
| **How it was built** | Hard-link Primitive skeleton → re-quantize fused expert `down_proj` from the abliterated BF16 → overlay the remaining residual writers in BF16. `gate_proj` / `up_proj` NVFP4 is Primitive's (OrcaRouter never touched `experts.gate_up_proj`). |

### Abliteration (from the OrcaRouter card)

Refusal-direction removal following Arditi et al. (2024). Residual-writing matrices only (149 tensors):

| Component | matrices edited |
|---|---|
| `self_attn.o_proj` (12 full-attention layers + MTP) | 13 |
| `linear_attn.out_proj` (36 GDN linear-attention layers) | 36 |
| `mlp.experts.down_proj` (fused, all 512 experts × 49 layers) | 49 |
| `mlp.shared_expert.down_proj` | 49 |
| `ple.value_proj` + `embed_tokens` | 2 |

**Never touched:** vision + video tower, MoE router, fused `experts.gate_up_proj`, Hyper-Connection mixers, QSA indexer, n-gram embedding table, `mtp.fc_*`, norms, `lm_head`.

Abliteration is a weight edit, not data-level unlearning. Fine-tuning on refusal-heavy / safety data can partially re-introduce refusals.

### Intended use

- Research into refusal mechanisms, alignment, and interpretability.
- Red-teaming and safety / robustness evaluation in controlled environments.
- A single-GPU NVFP4 serve of the uncensored Flash-Next weights.

OrcaRouter's BF16 eval (rule-based refusal classifier, not an LLM judge) is the capability / refusal reference for the parent; this graft does not re-run that suite. Sanity-checked on this checkpoint: text `pong`, vision OCR of a solid-red PNG, tool-calling parser flags identical to Primitive.

---

## Serve with vLLM

Stock image: `vllm/vllm-openai:qwen38-flash-next`. Two flags are **load-bearing** on one GPU:

- `--distributed-executor-backend mp` — without this the n-gram offload worker never starts; the server looks healthy and hangs on the first forward.
- `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800` — the worker loads a ~95 GB BF16 table; the 600 s default can expire first.

Do **not** use an FP8 n-gram table (RadixArk / Inferact-style). `VLLM_PLE_CPU_OFFLOAD` cannot load those.

Tool calls use Qwen3-Coder XML (`--tool-call-parser qwen3_coder`). Toggle thinking per request with `chat_template_kwargs={"enable_thinking": true|false}`. Pass `image_url` content parts for vision.

### NVIDIA GB10 / DGX Spark (this checkpoint's home)

Unified memory is **121 GiB**. Official “1× 96 GB GPU + ~100 GiB host RAM” is **two pools**; on GB10 they are the same pool, so the PLE table must page to a **~100 GiB swapfile** or the load OOMs. `--gpu-memory-utilization 0.75` (not 0.92). Hermes needs context ≥ 64K, so `--max-model-len 65536`.

```bash
# once per boot, before vLLM:
#   fallocate / swapon a ~100 GiB file (e.g. qwen38-ple.img)

docker run --name vllm --ipc=host --gpus all -p 8000:8000 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e CUTE_DSL_ARCH=sm_121a \
  -v /path/to/Qwen3.8-Flash-Next-Uncensored-NVFP4:/model \
  vllm/vllm-openai:qwen38-flash-next \
  /model \
  --served-model-name Qwen3.8-Flash-Next-Uncensored-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.75 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Discrete 96 GB Blackwell (Primitive's recipe)

Same image and PLE flags; raise utilization if GPU RAM and host RAM are separate:

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  vllm/vllm-openai:qwen38-flash-next \
  --model mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Host still needs ~100 GB free RAM (or swap) for the n-gram table.

---

## Measured on DGX Spark (GB10)

Host `edgexpert-3b5a`, 121 GiB unified memory, `vllm/vllm-openai:qwen38-flash-next`, PLE CPU offload, `--gpu-memory-utilization 0.75`, `--max-model-len 65536`, `--max-num-seqs 4`, ~100 GiB PLE swap. Same NVFP4 layout as Primitive; decode speed is the recipe, not the abliteration overlay.

### Closed-loop token sweep (thinking off)

`ignore_eos=true`, `max_tokens=256`, unique prefixes, 12 requests/level after 2 warmup. Closed-loop harness on this Spark, 2026-08-27, while Primitive NVFP4 was loaded (identical expert packing / PLE / vLLM flags).

| concurrency | output tok/s | TTFT p50 | TPOT p50 | notes |
|---|---:|---:|---:|---|
| 1 | **16.9** | 240 ms | 58.4 ms | isolated-user baseline |
| 2 | 29.7 | 433 ms | 66.3 ms | |
| 3 | 46.6 | 505 ms | 63.0 ms | |
| 4 | **57.0** | 472 ms | 67.9 ms | peak; `--max-num-seqs 4` |
| 5 | 41.9 | 999 ms (p95 19.2 s) | 70.4 ms | queues; TTFT blows up |

c=1 is ~17 tok/s / ~58 ms per output token. Do not compare to Primitive's 74 tok/s on a discrete RTX PRO 6000 — that card has a separate 96 GB GPU + host RAM. On GB10 the PLE table lives in NVMe-backed swap.

## What's quantized to what

| tensors | format |
|---|---|
| 48 layers × 512 routed experts (`gate` / `up` / `down`) | NVFP4 group 16 |
| PLE n-gram table (51B) | BF16 (required by `VLLM_PLE_CPU_OFFLOAD`) |
| attention, GDN, shared experts, routers, MTP, vision | BF16 |

MTP fused `experts.down_proj` stays BF16 (Primitive layout) and is overlaid from the abliterated source.

---

## Bias, risks, and limitations

- **Safety guardrails removed** — the model will produce harmful, biased, or offensive content on request (see the disclaimer).
- It inherits any biases and limitations of `Qwen3.8-Flash-Next` and of NVFP4 RTN (no calibration).
- OrcaRouter's reported refusal metric is a rule-based heuristic on the BF16 parent; this NVFP4 graft was not re-scored on AdvBench / HarmBench.
- On GB10, PLE paging to NVMe makes decode slower than a discrete 96 GB GPU with 100 GiB of spare host RAM.

## License

**[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)**, inherited from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Abliteration and NVFP4 packing do not change those obligations.

Credit: Qwen / Alibaba (base), [OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) (abliteration), [Primitive](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4) (single-GPU NVFP4 + BF16 PLE recipe).
