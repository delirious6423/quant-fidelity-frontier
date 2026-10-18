---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
library_name: transformers
tags:
- deepseek_v4
- moe
- nvfp4
- fp8
- quantized
- modelopt
- blackwell
- speculative-decoding
---

# DeepSeek-V4-Flash-0731 — NVFP4 routed experts (incl. MTP)

[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(revision `7872f01b`) with all **256 routed experts per MoE layer quantized to
NVFP4 W4A4** — including, since the 2026-08-03 update, the routed experts of
the **three MTP (multi-token-prediction) layers** — produced with NVIDIA
[Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)'s
DeepSeek-V4 routed-expert recipe. Everything else — attention, router gates,
shared experts, embeddings, and `lm_head` — is untouched from the source
release (FP8 per `quant_method: fp8`).

Built for Blackwell GPUs (sm_120+), where NVFP4 GEMMs run on native FP4 tensor
cores. Serves with vLLM / TensorRT-LLM builds that support the DeepSeek-V4
hybrid FP8 + NVFP4-MoE layout; verified under vLLM at TP=2 on
2× RTX PRO 6000 Blackwell (96 GB).

## 2026-08-03 update: NVFP4 MTP experts → working speculative decoding

The initial upload left the MTP layers' expert weights in the source MXFP4
format, which does not load correctly through the NVFP4 fused-MoE path — MTP
speculative-decoding acceptance collapsed to ~15% and spec decode was a net
loss. This update recasts the routed-expert weights of all 3 MTP layers to
NVFP4 with the same closed-form lossless cast: **2,304 tensors,
603,979,776 / 603,979,776 blocks lossless (100.0%), max dequant error 0.0**
(see `mtp_nvfp4_build_report.json`). Changed files: shards 46–48, the index,
`config.json`, and `hf_quant_config.json` (recipe version
`dsv4-nvfp4-experts-mtp-fallback`).

With this checkpoint MTP speculative decoding (DSpark) works, roughly doubling
single-stream decode throughput.

## Benchmarks

vLLM, TP=2 on 2× RTX PRO 6000 Blackwell (96 GB), PCIe (no P2P), 300 W/GPU
power cap, 131k context. 512-token greedy completions, temp 0, unique prompt
nonces; acceptance measured from vLLM `/metrics` counter deltas.

| Metric | MTP off | MTP spec decode (DSpark, 5 draft tokens) |
|---|---:|---:|
| Single-stream decode, median | 112.3 tok/s | **199.9 tok/s (1.78×)** |
| Single-stream decode, mean | 112.3 tok/s | **220.2 tok/s (1.96×)** |
| Single-stream range | 112.2–112.9 | 167–302 (content-dependent) |
| 2-stream aggregate | — | 353 tok/s |
| 4-stream aggregate | — | 411 tok/s |
| TTFT, ~1.8k-token prompt, streaming | — | 253–258 ms |

A second suite on the identical config measured median 242.5 / mean
232.7 tok/s — greedy tie-breaks change the generated text between sessions, so
single-stream medians move ±20% run to run. Typical: **~200–240 tok/s**.

Acceptance: mean acceptance length **3.85 tokens/step** (of 6 max), cumulative
draft acceptance **49.4%** — in line with DeepSeek's published DSpark figure
(50.8%). Throughput tracks acceptance roughly linearly (accept 2.8 →
~168 tok/s; accept 5.6 → ~302 tok/s); the spread is content predictability,
not jitter.

### Recommended spec-decode settings

- **Greedy draft sampling.** Probabilistic draft sampling garbles output
  through this NVFP4 path; greedy is clean at all target temperatures
  (verified at T = 0 / 0.7 / 1.0, including thinking mode).
- **5 draft tokens, fixed depth.** Depth 7 is slower (positions 6–7 accept
  only 1–10% and the verify cost is paid every step); the dynamic-depth
  controller oscillates at single-stream (166–177 tok/s).
- **bf16 draft head.** An FP8 draft head costs acceptance (3.51 vs 4.09
  accept length; 196 tok/s median).

## Why the weights are bit-exact

The source release already ships routed experts as **MXFP4** (E2M1 nibbles +
E8M0 power-of-two scale per 32-element block). Instead of the usual lossy
dequantize→requantize round trip, this conversion uses a closed-form cast:
`weight_scale_2` is pinned to `2^(k_max − 8)` and each 16-element block's E4M3
scale to `2^(k_j − m)`, so every NVFP4 nibble equals the source MXFP4 nibble
verbatim (w1/w3 share one `scale_2` for the fused GEMM1).

Result: **8,657,043,456 / 8,657,043,456** main-layer blocks and
**603,979,776 / 603,979,776** MTP-layer blocks lossless (100.0000%) — the
expert weights are byte-faithful to the original release. Quantization only
introduces the calibrated activation scales.

## Calibration

`input_scale` (per-tensor activation scale for W4A4) comes from post-training
calibration with Model-Optimizer:

- **Data**: 64 samples each from `abisee/cnn_dailymail` and
  `nvidia/Nemotron-Post-Training-Dataset-v2` (stem/chat/math/code splits)
- **Setup**: MP=2 over 2× RTX PRO 6000 Blackwell, batch size 4
- Experts that received no tokens during calibration fall back to the max
  observed scale among calibrated experts of the same projection; MTP-layer
  experts use the same fallback scheme

## Layout

Original HF 48-shard layout. Per routed-expert weight (main and MTP layers),
the MXFP4 `.scale` sibling is dropped and three keys are added:

| key | contents |
|---|---|
| `…weight` | NVFP4-packed uint8, shape `(out, in/2)` |
| `…weight_scale` | per-16-block E4M3 scale, shape `(out, in/16)` |
| `…weight_scale_2` | per-tensor FP32 scale |
| `…input_scale` | per-tensor FP32 activation scale |

`config.json` keeps the source FP8 `quantization_config` and adds
`moe_quant_algo: "NVFP4"` (group size 16); `hf_quant_config.json` carries the
full quantized-layer manifest (43 MoE layers + `mtp.{0,1,2}.ffn.experts`).
MTP shared-expert weights remain in the source format.

## Provenance

- Base: `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b` (304B params, 43
  layers, 256 routed experts, 3 MTP layers)
- Tooling: nvidia-modelopt 0.45.0, torch 2.13.0+cu130, safetensors 0.8.0
- License: MIT, inherited from the base model (see `LICENSE`)
