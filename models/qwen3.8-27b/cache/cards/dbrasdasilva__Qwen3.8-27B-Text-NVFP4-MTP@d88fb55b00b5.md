---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen3_5
  - qwen3.8
  - nvfp4
  - quantized
  - modelopt
  - mtp
  - speculative-decoding
  - blackwell
  - text-only
  - experiment
language:
  - en
  - zh
  - ja
  - ko
  - fr
  - de
  - es
  - it
  - pt
  - ru
  - ar
---

# Qwen3.8-27B-Text-NVFP4-MTP — an experiment

An **NVFP4-quantized, text-only** build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B): vision
tower removed, **MTP head grafted back in bf16** so vLLM speculative decoding works, exported in **`modelopt`**
format for the native NVFP4 path on NVIDIA Blackwell / SM120.

## Update, 2026-10-02 — successors, and the same question on 96 GB

This build was an experiment on a **32 GB** RTX 5090, and its conclusions still apply
there. Two things have changed since.

**There are now vision-kept successors**, quantized from the bf16 base with the vision
tower *and* the MTP head intact:

- [`dbrasdasilva/Qwen3.8-27B-NVFP4-MTP`](https://huggingface.co/dbrasdasilva/Qwen3.8-27B-NVFP4-MTP) — mixed FP8/NVFP4, 24 GB
- [`dbrasdasilva/Qwen3.8-27B-FP8-MTP`](https://huggingface.co/dbrasdasilva/Qwen3.8-27B-FP8-MTP) — FP8 throughout, 30 GB

Both also carry **calibrated FP8 KV-cache scales** (`k_scale`/`v_scale`), which this
build does not. Without them `--kv-cache-dtype fp8` runs at scale 1.0 and loses
accuracy silently — vLLM says so in the log, and it is easy to miss.

**On 96 GB the central trade-off disappears.** The "pick two" conclusion below is a
property of a 32 GB KV budget, not of the model. Measured on an RTX PRO 6000 Blackwell
96 GB with vLLM 0.30.0, vision on, MTP on, `--max-model-len 262144`,
`--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.95`:

| Checkpoint | Weights | KV cache | Concurrency at 262 k |
|---|---|---|---|
| `Qwen3.8-27B-NVFP4-MTP` (mixed FP8/NVFP4) | 24 GB | 1,796,931 tokens | 6.85× |
| `Qwen3.8-27B-FP8-MTP` | 30 GB | 1,622,656 tokens | 6.19× |
| `unsloth/Qwen3.8-27B-NVFP4` | 22 GB | 1,820,363 tokens | 6.94× |
| `nvidia/Qwen3.8-27B-NVFP4` | 20 GB | 2,118,244 tokens | 8.08× |

Vision, MTP and the model's native 262 k context coexist with room to spare, so the
32 GB question — *does stripping vision buy context or speed?* — stops being
interesting rather than getting a different answer. (Decode throughput was not
re-measured here; those are KV-cache ceilings, not tok/s.)

**One hard-won detail if you quantize this architecture yourself.** A build that
quantized `lm_head` while still listing it in `quantization_config.ignore` would not
load at all:

    ValueError: There is no module or parameter named 'lm_head.input_scale' in
    Qwen3_5ForCausalLM ... are: {'lm_head.weight'}

It is not that a quantized `lm_head` is unsupported — `nvidia/Qwen3.8-27B-NVFP4` ships
an NVFP4 one and loads fine. The config simply has to tell the truth about the weights.
This build keeps `lm_head` in bf16 and says so, which is why it was never affected.

## TL;DR
This repo exists to answer one question: *on a 32 GB Blackwell card (RTX 5090), does physically stripping the
vision tower give you more context or more speed?* **It doesn't** — it only removes the **921 MB** vision tower
from the *download* (nothing at runtime), because vLLM already skips that tower at load when you pass
`--language-model-only`. Note this build also grafts back an **849 MB** bf16 MTP head, so the net file is only
~0.95 GB smaller than a build that carries *both* vision and MTP (e.g. the gittensor build below).

**Recommendation: on 32 GB, run the full multimodal build instead** and toggle MTP for what you need —
**MTP off → bigger KV cache (~199K context, vision on)**, **MTP on → less context (~123K) but ~1.6× the speed**.
Pure-text on the multimodal build (`--language-model-only`) is *identical* to this stripped build, so there's
rarely a reason to prefer this one. It's published for reproducibility and for the ~1 GB smaller download.

## The experiment
- **Base / method**: `Qwen/Qwen3.8-27B` → `nvidia-modelopt` 0.43.0 NVFP4 (`NVFP4_DEFAULT_CFG`, group_size 16),
  keeping `lm_head`, `*linear_attn.conv1d*`, `mtp.*`, and `model.visual.*` in bf16; then the vision tensors are
  **physically deleted** and the 15 bf16 `mtp.*` tensors grafted back so speculative decoding engages.
- **Hardware / engine**: one **RTX 5090 (32 GB)**, **vLLM 0.27.2** (needs the ≥0.27.2 gated-delta-net fix; older
  builds crash MTP at long context), `--gpu-memory-utilization 0.93`, fp8 KV cache, `--max-num-seqs 1`.
- **Important condition — this was a *shared* card, not a dedicated one.** The desktop was live on the same 5090
  (~2 GB of VRAM held by the OS compositor) during every run, and vLLM's cuda-graph profiler over-reserves
  another ~0.4 GB unless disabled. So the numbers below are **conservative, shared-card results**, taken with
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` set to reclaim that 0.4 GB. A card **fully dedicated to inference**
  (desktop moved to an integrated GPU, ~2 GB freed) does strictly better — more KV cache, higher context ceilings.
- **What we compared**: this text-only build vs. a full multimodal NVFP4 build of the same model, across
  vision-on/off and MTP-on/off, measuring the KV-cache ceiling and steady-state decode tok/s.

## Results
| Serve mode (multimodal build unless noted) | Vision | Max context¹ | Decode tok/s |
|---|---|---|---|
| vision on, **MTP off** (drop `--speculative-config`) | ✅ | **~199K** | ~77 |
| vision on, **MTP on** | ✅ | ~123K | ~125–135² |
| `--language-model-only`, MTP on (text) | — | ~200K | ~130–140 |
| *this* text-only build, `--language-model-only`, MTP | — | ~200K | ~130–140 |

¹ **Every ~200K/199K figure is conditional on `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** (see below);
without it you top out at **~185–188K**.
² An image in the prompt does **not** slow decode — the vision encoder runs once during prefill (measured ~133
tok/s decoding *with* an image present). Time-to-first-token grows with image resolution; steady decode doesn't.

**What each row taught us:**
1. **Stripping vision ≠ more context or speed.** Text-only build and multimodal build, both `--language-model-only`
   + MTP, are identical: **KV 201,428 tokens, ~130–140 tok/s each.** Deleting the tower only shrinks the download.
2. **The 200K lever is a vLLM env var, not the model.** vLLM 0.27.x's cuda-graph profiler over-reserves ~0.4 GB
   (estimates 0.40 GB, actually uses 0.04 GB). `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` hands it back
   (KV 6.95 → 7.35 GB) → full 200K fits at util 0.93. Applies to any NVFP4 27B on 32 GB. *(Context: a normal
   desktop compositor already holds ~2 GB of the 5090; dedicate the card — desktop on an iGPU — and you can push
   past 200K.)*
3. **MTP, not vision, is what eats KV cache.** The vision tower costs ~0.17 GB of weights (≈nothing). Enabling
   MTP is what roughly halves usable context (199K → 123K with vision on).
4. **Vision works and is nearly free.** The multimodal build correctly read shapes + text from a test image.
5. **On 32 GB you can't have vision + MTP-speed + 200K at once — pick two.** The scarce resource is KV cache.

## Conclusion
Prefer the **multimodal build** on 32 GB: it's a strict superset (does everything this build does, plus vision)
and you choose context-vs-speed by toggling MTP. Reach for **this text-only build only if** you never send images
and want the ~0.95 GB smaller download or a guaranteed text-only artifact.

**Two conditions behind every number here:** they were measured on a **shared** card (the desktop held ~2 GB of
the 5090) with **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** set to reclaim vLLM's ~0.4 GB profiler
over-reservation. Without that env var you lose ~12–15K of context (~185–188K instead of 200K); dedicate the card
(desktop on an iGPU, ~2 GB freed) and every context ceiling above rises further.

## Serving this build (vLLM ≥ 0.27.2, Blackwell / SM120)
```bash
# The env var is REQUIRED to reach 200K on a 32 GB card. Without it you get ~185–188K.
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
vllm serve dbrasdasilva/Qwen3.8-27B-Text-NVFP4-MTP \
    --trust-remote-code \
    --quantization modelopt \
    --language-model-only \
    --kv-cache-dtype fp8 \
    --max-model-len 200000 \
    --max-num-seqs 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

> **`--language-model-only` is required for this build.** The vision tower and its `preprocessor_config.json` were
> removed, but the arch class is still `Qwen3_5ForConditionalGeneration` — without the flag vLLM tries to build an
> image processor that no longer exists and aborts with `Can't load image processor for '…'`.

## Model details & credit
- **Quant**: `nvidia-modelopt` 0.43.0 NVFP4, group_size 16, KV-cache quant off; kept in bf16: `lm_head`,
  `model.visual.*` (then deleted), `*linear_attn.conv1d*` (Mamba-style SSM convs), `mtp.*` (grafted back).
  Calibration: 20 samples from `neuralmagic/calibration` (LLM split), max_seq_len 8192.
- **Vision removal (how)**: after the ModelOpt NVFP4 export, the `model.visual.*` tensors are dropped from every
  `.safetensors` shard and from `model.safetensors.index.json`, and `vision_config` + the image/video/vision-start
  token ids are removed from `config.json` (and `preprocessor_config.json` deleted), using the **`safetensors`**
  library. The approach follows **lna-lab's** GGUF-to-NVFP4-SM120 recipe (credited below).
- **Working MTP matters**: naive `AutoModelForCausalLM` export drops the MTP head → 0% draft acceptance; the
  ignore pattern must be `mtp*` (not `model.mtp.*`, which never matches). With it, ~72% acceptance measured.
- **Size accounting (measured)**: core weights ~18.79 GB; the removed vision tower is **921 MB** (333 bf16
  tensors); the grafted MTP head is **849 MB** (bf16). Because those two nearly cancel, this build is only
  ~0.07 GB smaller than a multimodal build *without* MTP, but ~0.95 GB smaller than one *with* both vision and
  MTP (the gittensor build: 20.59 GB vs this 19.64 GB). The download saving is entirely the vision tower.

This is a **derivative work** — all capability belongs to its authors; this repo only re-quantizes and strips:
1. **Qwen team (Alibaba)** — [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
2. **osoleve** — the original text-only NVFP4 + MTP-graft recipe this descends from,
   [`osoleve/Qwen3.5-27B-Text-NVFP4-MTP`](https://huggingface.co/osoleve/Qwen3.5-27B-Text-NVFP4-MTP).
3. **lna-lab** — the reproducible pipeline, [`lna-lab/GGUF-to-NVFP4-SM120`](https://github.com/lna-lab/GGUF-to-NVFP4-SM120).
4. **sakamakismile** — the Qwen3.6 build this mirrors,
   [`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP).
