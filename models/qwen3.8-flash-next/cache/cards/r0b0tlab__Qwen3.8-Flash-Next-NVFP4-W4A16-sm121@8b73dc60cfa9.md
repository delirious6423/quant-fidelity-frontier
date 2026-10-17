---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
  - base_model:Qwen/Qwen3.8-Flash-Next
  - base_model:quantized:Qwen/Qwen3.8-Flash-Next
  - nvfp4
  - w4a16
  - modelopt
  - sglang
  - nextn
  - dgx-spark
  - sm121
---

# Qwen3.8-Flash-Next NVFP4 W4A16 — 2× GB10 / SM121

Self-quantized **ModelOpt NVFP4 W4A16** checkpoint for
**Qwen3.8-Flash-Next**, qualified with SGLang on two NVIDIA GB10 systems using
TP=2 over RoCE.

- Base model: `Qwen/Qwen3.8-Flash-Next` revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Checkpoint: 131 safetensors shards
- Quantization: NVFP4 W4A16 weights, BF16 activations, unquantized BF16 `mtp.*` NEXTN tensors
- Production KV: FP8 E4M3 target and draft caches
- Checkpoint audit: **PASS** — 1,851 tensors, 31 BF16 `mtp.*` tensors, zero preserved-digest mismatches
- Runtime package: https://github.com/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang
- Runtime image: `ghcr.io/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang@sha256:9d7406a4f2b5d829c2646e8408ce0d02ccd20d7b98f5d7a3198af3c6a81854e0`

## Qualified profiles

- SGLang source: `7d2e8fcb9a768d0dc7e21d56be293680ed10a106`
- Runtime image ID: `sha256:49d06b35dbebddb379294941e390e9e96f7e7a86815fa741278474c5ce0055be`
- Context: 262,144 tokens
- Max running requests: 4
- NEXTN: 3 steps, 4 draft tokens, top-k 1
- Decode CUDA-graph buckets: `[1,2,3,4]`
- FP8 E4M3 target/draft KV, FP32 GDN SSM state
- Chunk/max-prefill: 1,024 tokens
- Memory fraction: 0.90
- Weight loading: mmap disabled

Quality was measured on the strict full-context AR profile (profile SHA-256
`ea736a4837d844adb95d92d56ba4edf23c3c620533c76fadad2b26882a57af2a`).
The public click-run uses the qualified full-context NEXTN profile (profile
SHA-256 `5a2ec22568aab93707e3fe44aa848a4c3648365f4c812738c746c033efaad401`).

## Results

### Gate summary

| Gate | Result |
|---|---|
| Full-context runtime | **PASS** |
| NIAH | **PASS** |
| Q200-v2 scoring completeness | **PASS** |

### Q200-v2 — native thinking, low effort

| Family | Correct | Incorrect | Total |
|---|---:|---:|---:|
| GSM8K | **80** | 0 | 80 |
| HumanEval | **39** | 1 | 40 |
| IFEval | **36** | 4 | 40 |
| Hard reasoning | **20** | 0 | 20 |
| BFCL v4 `multi_turn_base` structural-hard20 | **13** | 7 | 20 |
| **Overall** | **188** | **12** | **200** |

**Q200-v2: 188/200 (94.0%).** All 200 responses were transported and
explicitly graded under the frozen native-thinking profile.

`BFCL v4 multi_turn_base structural-hard20` is a frozen, model-independent
20-case structural-hard subset scored with the official `bfcl-eval` partial
evaluator (`2025.12.17`). It is not a full-category BFCL score and is not a
claim about the “20 hardest” cases.

### Optimized short-request concurrency

Method: 1,024 input tokens → 256 output tokens; aggregate output throughput on
the promoted graph-on NEXTN s3d4 profile.

| Concurrency | Median output tok/s |
|---:|---:|
| 1 | **63.84** |
| 2 | **83.66** |
| 4 | **132.64** |

Dedicated C1 median: **62.10 output tok/s**. Zero request errors and zero
graph-fallback delta.

### NIAH

| Gate | Result |
|---|---|
| NIAH | **PASS** |

## Click-run

Use the paired-node click-run in the public runtime repository. Clone the repo
on both arm64 GB10 nodes and use one shared epoch.

```bash
git clone https://github.com/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang.git
cd qwen38-flash-next-w4a16-sm121-sglang
export CAMPAIGN_EPOCH="clickrun-$(date -u +%Y%m%dT%H%M%SZ)"
```

Rank 0:

```bash
RANK=0 DIST=<rank0-roce-address>:20000 PEER_HOST=<rank1-ssh-host> \
IFACE=<rank0-roce-interface> IBHCA=<rank0-hca-device> \
CAMPAIGN_EPOCH="$CAMPAIGN_EPOCH" bash scripts/click_run_tp2.sh
```

Rank 1, with the same epoch:

```bash
RANK=1 DIST=<rank0-roce-address>:20000 PEER_HOST=<rank0-ssh-host> \
IFACE=<rank1-roce-interface> IBHCA=<rank1-hca-device> \
CAMPAIGN_EPOCH="$CAMPAIGN_EPOCH" bash scripts/click_run_tp2.sh
```

The script downloads this model at an immutable Hugging Face revision,
verifies all 131 shard checksums, pulls the immutable runtime image by digest,
checks the image config ID, launches the frozen full-context profile, and
gates readiness plus native-thinking semantic warmup. Set `MODEL_DIR` to reuse
an already verified checkpoint and `PEER_SSH_IDENTITY_FILE` if SSH uses a
non-default key.

## Limitations

- Qualified runtime: **2× GB10 / SM121, arm64, TP=2**. Do not infer x86_64 or
  SM120 performance from these measurements.
- The quality claim is text/tool-calling only; multimodal/video quality is not
  included.
- The BFCL result is only the explicitly labeled 20-case subset.
- The measured W4A16 path uses Marlin on SM121.

## License and credit

Weights are derived from `Qwen/Qwen3.8-Flash-Next` and remain subject to the
Qwen license terms (`license: other`; see the base model card). Runtime
packaging and scripts are MIT licensed. Credits: Qwen, SGLang, NVIDIA
ModelOpt, BFCL contributors, RadixArk, and the upstream contributors identified
in the runtime repository.
