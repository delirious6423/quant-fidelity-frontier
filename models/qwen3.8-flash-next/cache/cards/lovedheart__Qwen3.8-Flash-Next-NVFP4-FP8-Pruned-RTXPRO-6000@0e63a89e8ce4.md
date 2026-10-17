---
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
license: other
library_name: Model Optimizer
tags:
- ModelOpt
- Qwen3.8
- quantized
- pruned
- FP4
- fp4
- FP8
- fp8
- mixed-precision
- blockwise
- sglang
---

> ## ✂️ This is a PRUNED model — 512 → 448 experts per layer (AIMER)
>
> This checkpoint is a **pruned version of Qwen3.8-Flash-Next**: the routed
> MoE experts were reduced from **512 to 448 per layer** (64 experts removed
> from each of the 48 MoE layers, K=64) using the **AIMER** method
> ([AIMER: Calibration-Free Task-Agnostic MoE Pruning](https://arxiv.org/abs/2603.18492)
> — *A*bsolute mean over root mean square *IM*portance for *E*xpert *R*anking).
> No calibration data or activations are needed: experts are scored per
> layer by `mean|W| / RMS(W)` and the lowest-scoring 64 in each layer are
> pruned; remaining experts are contiguously re-indexed and the router
> (`gate`) rows are kept aligned. The MTP layer is pruned to 448 as well, so
> `--speculative-algo NEXTN` remains usable.

**Serve with one Docker command** (image bundles the required patched SGLang;
download this repo locally first, mount it at `/models`; container port 30000
by default, map any host port; all `launch_server` flags pass through after the
image name). `latest` adds the **QSA FP8 indexer** — set
`SGLANG_QSA_USE_FP8_INDEXER=1` to keep **512K-context prefill fast**:

```sh
docker run --rm --gpus all --ipc=host -p 8070:30000 \
  -e SGLANG_QSA_USE_FP8_INDEXER=1 \
  -v ~/sglang-cache:/root/.cache/sglang \
  -v /path/to/this/repo:/models:ro \
  lovedheart/qwen38-flash-next:latest \
  --served-model-name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 30000 --trust-remote-code \
  --tensor-parallel-size 1 --max-running-requests 4 --chunked-prefill-size 2048 \
  --mem-fraction-static 0.94 \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy \
  --kv-cache-dtype fp8_e4m3 --max-prefill-tokens 8192 \
  --schedule-policy lpm --allow-auto-truncate \
  --mm-feature-transport=cpu --enable-cache-report --enable-metrics \
  --sleep-on-idle --max-mamba-cache-size 10 \
  --sampling-backend flashinfer --enable-session-radix-cache --ple-offload-embedding \
  --moe-runner-backend flashinfer_cutlass \
  --reasoning-parser auto --attention-backend flashinfer --linear-attn-decode-backend flashinfer \
  --mm-attention-backend flashinfer_cudnn --tool-call-parser auto \
  --speculative-algo NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec
```

> ## ⚠️ Requires a patched SGLang — stock builds emit GARBLED OUTPUT
>
> This is a **mixed-precision** checkpoint: NVFP4 routed experts **plus FP8
> E4M3 2D-blockwise (128×128) weight-only attention projections**
> (`self_attn.q/k/v/o_proj` on the 12 QSA layers, and `linear_attn.in_proj_qkv
> / in_proj_z / out_proj` on the 36 GDN layers). These FP8 layers are marked
> `quant_algo: "FP8_PB_WO"` in `config.json`'s `quantized_layers` map.
>
> **Stock SGLang does not dispatch `FP8_PB_WO`.** `ModelOptMixedPrecisionConfig`
> only handles `FP8 / MXFP8 / NVFP4 / W4A16_NVFP4`; `FP8_PB_WO` layers fall
> through to `UnquantizedLinearMethod`, which loads the packed FP8 bytes into
> BF16 parameters **silently — the server starts without any error and produces
> garbage text.**
>
> Run it on **https://github.com/lovedheart/sglang/tree/feat/qwen38-flash-next**
> (based on [sgl-project/sglang#36497](https://github.com/sgl-project/sglang/pull/36497)
> at `73a2552`). To apply just the fix to a tree based on `73a2552`:
>
> ```bash
> git remote add lovedheart https://github.com/lovedheart/sglang.git
> git fetch lovedheart feat/qwen38-flash-next
> git cherry-pick 4638f9e   # modelopt_mixed: dispatch FP8_PB_WO -> Fp8LinearMethod([128,128])
> ```
>
> Startup check: the log must show `quant=modelopt_mixed` and **no** missing-key
> warnings for `*.weight_scale_inv`. If you see either, the FP8 layers are not
> being read and output will be corrupt.

# Model Overview

## Description:
RadixArk Qwen3.8-Flash-Next-NVFP4 is the quantized version of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
a hybrid-architecture multimodal Mixture-of-Experts model. Quantization was
performed with [NVIDIA Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
(snapshot `87c9f8cf`) using the NVFP4 W4A4 recipe, scoped to the routed
experts only.

This checkpoint is a private candidate release.

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer

### License/Terms of Use:
See the [source model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for
license terms.

### Use Case:
Developers evaluating NVFP4-quantized serving of Qwen3.8-Flash-Next for
agentic systems, chat, coding, and multimodal reasoning workloads.

### Release Date:
Hugging Face on 08/25/2026 via https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4

## Model Architecture:
**Architecture Type:** Transformer (hybrid GDN + QSA sparse attention,
multi-hyperconnection streams, PLE n-gram injection) <br>
**Network Architecture:** Multimodal MoE — 48 decoder layers, 512 routed
experts per MoE layer (top-10 routing) + shared expert, 1 MTP layer <br>
**Number of Model Parameters:** ~180B in total (360 GB BF16 source) <br>

## Input:
**Input Type(s):** Text, Image, Video <br>
**Input Format(s):** String, RGB, Video <br>
**Other Properties Related to Input:** Context length up to 262K <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* SGLang (with `qwen4_exp` model support) <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell (validated on GB300 and B300) <br>

**Preferred Operating System(s):** <br>
* Linux <br>

## Model Version(s):
NVFP4 candidate 1.0, quantized with nvidia-modelopt v0.46.0 (snapshot
`87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c`).

## Calibration Dataset:
**Link:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
(config 3.0.0, train split) <br>
**Properties:** 128 articles (seed 1234), truncated to 512 tokens; MoE-block
input activations captured from live SGLang serving (prefill only),
62,139 rows per layer; activation scales by max calibration over 8 seeded
sampled batches per part. Representativeness probe: GSM8K train [0:16] x2. <br>

## Post Training Quantization
This model was obtained by quantizing Qwen3.8-Flash-Next to NVFP4, ready for
inference with SGLang. **Only the routed experts of the 48 main-model MoE
layers are quantized** (fused `gate_up_proj` / `down_proj`; 294,912 quantized
tensor entries) to NVFP4 W4A4 (E2M1, group size 16, FP8 E4M3 block scales,
FP32 global scales, dynamic NVFP4 activations). Attention, QSA, GDN, mHC,
shared experts, routers, embeddings, LM head, vision, and **all 31 MTP
tensors remain BF16 and byte-identical to the source**. **The PLE n-gram
embedding tables use the FP8-quantized versions from the updated
`Qwen/Qwen3.8-Flash-Next-FP8` revision** (128 shards F8_E4M3 + per-table
scalar scale, dequantized to BF16 at load time); the remaining PLE weights
stay BF16. No KV-cache quantization metadata. Checkpoint size is reduced
from 360 GB to 135 GB (~2.7x).

## Usage

Serve with SGLang (requires a build with `qwen4_exp` support):

```sh
python -m sglang.launch_server \
  --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --tp 2 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-track-interval 64 \
  --chunked-prefill-size 4096 \
  --max-running-requests 36 \
  --context-length 262144 \
  --mem-fraction-static 0.80 \
  --allow-auto-truncate \
  --port 30000
```

## Evaluation
Results attributed to this exact checkpoint unless noted otherwise:

| Eval | Protocol | BF16 reference¹ | NVFP4 (this checkpoint) |
|---|---|---|---|
| GSM8K | full 1319, t0.6 / top-p 0.95 / max 8192 | 97.12–97.50² | **97.27** (stop 98.86, err 0) |
| AIME26 | 30 problems x 8, t1.0 / max 130k | 100 (240/240) | **98.75 pass@1** (majority@8 100, stop 99.17) ³ |

¹ BF16 reference runs were recorded on an earlier checkpoint revision of the
same model line; the two revisions' weight deltas are not established, so
treat comparisons as indicative, not exact.
² Range across three independent BF16 runs on the earlier revision.
³ Measured on the previous revision of this checkpoint, whose weights differ
from the current revision only in the PLE embedding tables.

**Behavioral note:** this quantization preserves single-turn accuracy
(GSM8K/AIME in-band); long agentic generations tend to run longer than BF16.

## Integrity Evidence
All audits pass on this exact checkpoint: structural audit (294,912 routed +
1,562 unchanged + 31 MTP tensors), scale audit (221,184 finite positive
scales, min 2.13e-05, max 448.0), unchanged-content byte-equality audit
(1,562 tensors / 118.4 GB), deterministic serving smoke. Raw metrics:
`gsm8k_metrics.json`, `aime26_metrics.json`;
audit reports and details: `qualification-notes.md`,
`validate_checkpoint_report.json`, `validate_scales_report.json`,
`audit_unchanged_report.json`.

## Model Limitations:
The base model was trained on data that contains toxic language and societal
biases originally crawled from the internet. Therefore, the model may amplify
those biases and return toxic responses especially when prompted with toxic
prompts. The model may generate answers that may be inaccurate, omit key
information, or include irrelevant or redundant text producing socially
unacceptable or undesirable text, even if the prompt itself does not include
anything explicitly offensive.
