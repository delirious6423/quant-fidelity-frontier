---
license: apache-2.0
language:
  - en
  - zh
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - gptq
  - int4
  - vllm
  - xpu
  - intel-arc
  - mtp
  - gdn
---

# Qwen3.8-27B — GPTQ INT4 (sym, G128) with preserved BF16 MTP head

<p align="left">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="license">
  <img src="https://img.shields.io/badge/quant-GPTQ_INT4_sym_G128-blue" alt="quant">
  <img src="https://img.shields.io/badge/engine-vLLM_XPU-purple" alt="engine">
  <img src="https://img.shields.io/badge/tested-Intel_Arc_Pro_B70_32GB-teal" alt="hardware">
  <img src="https://img.shields.io/badge/spec--decoding-MTP4-ff6f00" alt="mtp">
  <img src="https://img.shields.io/badge/vision-included-pink" alt="vision">
</p>

Self-quantized on the target GPU (gptqmodel 7.3.2) for **vLLM XPU serving on
Intel Arc (Xe2)**. The single MTP draft layer is deliberately **excluded from
quantization and kept in BF16**, so native speculative decoding (MTP) works
out of the box — and an optional **runtime INT4 draft overlay** (below) takes
single-stream decode from **83.7 → 112.7 tok/s** on the same checkpoint.

> All numbers are **real measurements** taken on a single Intel Arc Pro B70
> 32 GB — client-side timing, n=5 medians, 230 W cap, context 131072, fp8 KV,
> served at the Qwen3.8-27B model card's recommended sampling —
> **non-thinking: temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5**
> (thinking: 1.0 / 0.95 / 20). Greedy rows are marked greedy.
> Commands, methodology and raw logs are public: 📖
> **[Intel Arc Pro B70 Inference Cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)** (see the link section below).

## TL;DR

| What | Number | Notes |
|---|---:|---|
| Best single-stream decode (MTP4 + INT4 draft overlay, cache off) | **112.65 tok/s** | p512/g128, n=5 · [record](https://www.localmaxxing.com/en/runs/cmszpqy000e8fms014ty6i5x3) |
| Same, prefix cache on | **106.7 tok/s** | n=5, 103.2–111.3 · [record](https://www.localmaxxing.com/en/runs/cmt03mj040eh8ms01trjvhm75) |
| Checkpoint as-is (BF16 draft, MTP4) | **83.7 tok/s** | p512/g128, n=5 · [record](https://www.localmaxxing.com/en/runs/cmsur82fz06svms01ga1f0z83) |
| Cold input | ~1,700–1,850 tok/s | prompt tokens / TTFT, uncached |
| 5 concurrent coding users (8K sessions) | **25.5 tok/s each** (Σ 127.4) | 60/60 OK · [record](https://www.localmaxxing.com/en/runs/cmt03mjo60ehbms0117c5i745) |
| Short-prompt concurrent aggregate | **203.8 (C5) / 224.2 (C32) tok/s** | lmx harness · [C5](https://www.localmaxxing.com/en/runs/cmt00hzwf0effms014mdyizca) · [C32](https://www.localmaxxing.com/en/runs/cmt00i0eb0eflms012anb0yau) |
| Full-context (130944+128 = 131,072 tokens) | completes, 56–63 tok/s | single-request capacity check |
| C1 at recommended sampling (temp 0.7 / top_p 0.8 / k 20) | **102.6 tok/s** | 91.8% MTP acceptance · [record](https://www.localmaxxing.com/en/runs/cmt051y730eiims01prf8laxc) |
| 5 coding users at recommended sampling | **21.6 tok/s each** (Σ 107.8) | [record](https://www.localmaxxing.com/en/runs/cmt051ytx0eilms01fdxrujeh) |

## 📖 Cookbook — setup, patches, methodology, evidence

- **Repo:** [intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- **This model's recipe:** [QWEN38-VLLM-XPU.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/qwen38-27/QWEN38-VLLM-XPU.md)
- **INT4 draft overlay:** [DRAFT-INT4-S-M1.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/qwen38-27/DRAFT-INT4-S-M1.md)
- **Copy-paste launch lines:** [FULL-SETUP-COMMANDS.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/FULL-SETUP-COMMANDS.md)
- **Image + patch matrix:** [IMAGE-AND-PATCH-MATRIX.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/IMAGE-AND-PATCH-MATRIX.md)

## Model facts

| Property | Value |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` (hybrid GDN: linear attention + full attention every 4 layers) |
| Hidden / layers / heads | 5120 / 64 / 24 Q + 4 KV (GQA) |
| Backbone dtype | float16, ~18.2 GiB (5 safetensors, 2,399 tensors) |
| Vision tower | F16, unquantized, ~0.86 GiB — **included** (see below) |
| Max position embeddings | 262,144 (config value; measured serving on 32 GB is 131,072) |

## Quantization contract

- **GPTQ INT4, symmetric, group size 128, `desc_act=false`**, packed int32
- `quantization_config.dynamic` excludes `-:.*mtp.*` → **15 `mtp.*` draft
  tensors stay BF16** (this is what keeps native MTP loadable and reversible)
- Per-tensor error log ships as `quant_log.csv`
- Runs through the XPU W4A16 kernel family (`--quantization gptq`)

## Single-stream performance (C1, MTP4 unless noted, cache off, BF16 draft)

Stack: `vllm/vllm-openai-xpu@sha256:f01e24f6…` (vLLM 0.27.2rc1.dev77,
vllm-xpu-kernels 0.1.12.3), fp8 KV, scheduler 8192, 230 W cap, n=5.

| Cell | no-spec | MTP1 | MTP2 | MTP4 |
|---|---:|---:|---:|---:|
| Cold input p8192/g1 (tok/s) | 1774 | 1738 | 1736 | 1728 |
| Decode p512/g128 | 32.9 | 52.0 | 65.8 | **83.7** |
| Decode p8192/g128 | 31.5 | 50.0 | 62.9 | **77.1** |
| Full-context p130944/g128 | 23.2 | 38.9 | 44.4 | **56.3** |

## Optional: INT4 draft overlay (+39% decode, same checkpoint)

Two runtime patches quantize only the **draft side** (draft LM head + 5 MTP
linears) to INT4-G128 at server start; the target body and verify head stay
GPTQ-INT4/BF16. Off by default — the checkpoint on this hub is unchanged.

```bash
-e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1   # + the two draft patches
```

| Cell (n=5) | BF16 draft (as-is) | INT4 draft overlay | Δ |
|---|---:|---:|---:|
| p512/g128, cache off | 81.2 (matched arm) | **112.65** | +38.7% |
| p8192/g128, cache off | 77.5 | **103.6** | +33.7% |
| p512/g128, prefix on | — | **106.7** | — |
| MTP acceptance | 95.9% | 94.4% | −1.5 pp |

This is a **speed option, not a parity claim**: draft logits differ. On a
15-task temperature-0 suite both arms score 12/15 with zero regressions
unique to the overlay. Details, patches and hashes:
[DRAFT-INT4-S-M1.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/qwen38-27/DRAFT-INT4-S-M1.md).

## Concurrent serving (requires the v5 mixed-split patch)

Unpatched `gdn_attention` on this XPU stack dies on mixed speculative +
non-speculative batches at C≥2; the cookbook's `patch_gdn_mixed_split_v5.py`
fixes it (0 crashes across all cells below, including mixed long-prefill +
decode).

| Workload | Per-user | Total | TTFT p50 |
|---|---:|---:|---:|
| 5 concurrent coding sessions (~8K context, 3 turns, 512-token answers) | **25.5 tok/s** | 127.4 | 22–25 s |
| Same, first (cold) turn | 33.1 | 165.5 | 22.6 s |
| 5 short-prompt streams (lmx harness) | ~40 | **203.8** | 0.4 s |
| 32 short-prompt streams | ~28 | **224.2** (Σ-streams 903) | 15.4 s |

Two honest caveats (measured, open issues on this build): MTP acceptance
drops to 43–56% under concurrency, and **prefix-cache reuse across
concurrent sessions mostly fails** (0–38% hit rate vs 91% single-stream), so
warm multi-user turns re-prefill and TTFT stays 20+ s.

## Quick start

Full copy-paste recipe (image digest, patch hashes, launch lines for every
configuration): **[FULL-SETUP-COMMANDS.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/FULL-SETUP-COMMANDS.md)**.

<details>
<summary>Minimal text-only MTP4 launch (click to expand)</summary>

```bash
docker pull vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
docker run -d --name qw38 -p 8000:8000 --device /dev/dri \
  -v /path/to/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  -v $COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro \
  -v $COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  --entrypoint bash vllm/vllm-openai-xpu@sha256:f01e24f6… -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model \
   --quantization gptq --dtype float16 --max-model-len 131072 \
   --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 \
   --max-num-seqs 64 --max-num-batched-tokens 8192 \
   --served-model-name qwen38 --language-model-only \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'
```
</details>

## Vision is included — no extra files

The checkpoint ships the full F16 vision tower (333 `model.visual.*` /
`model.merger.*` tensors). To serve vision: keep `preprocessor_config.json`
and `video_preprocessor_config.json` (both in this repo) next to the
safetensors and drop `--language-model-only`. Measured on B70: 3,219 MiB
free after load with full 131K context retained; an image+text request
(~1,070 prompt + 200 output tokens) completes in 7.3 s.

## References

- **Cookbook (setup, patches, methodology, full evidence):**
  [intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) —
  start at [QWEN38-VLLM-XPU.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/qwen38-27/QWEN38-VLLM-XPU.md)
- **LocalMaxxing records:** [83.7](https://www.localmaxxing.com/en/runs/cmsur82fz06svms01ga1f0z83) ·
  [112.65](https://www.localmaxxing.com/en/runs/cmszpqy000e8fms014ty6i5x3) ·
  [106.7](https://www.localmaxxing.com/en/runs/cmt03mj040eh8ms01trjvhm75) ·
  [C5 realistic](https://www.localmaxxing.com/en/runs/cmt03mjo60ehbms0117c5i745) ·
  [C5 203.8](https://www.localmaxxing.com/en/runs/cmt00hzwf0effms014mdyizca) ·
  [C32 224.2](https://www.localmaxxing.com/en/runs/cmt00i0eb0eflms012anb0yau)
  (accepted leaderboard records)

## Provenance & license

Converted locally from the official `Qwen/Qwen3.8-27B` release with
gptqmodel 7.3.2. Apache-2.0, following the base model's license. The MTP
draft tensors are bit-identical to the source release.
