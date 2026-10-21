---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: sglang
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - nvfp4
  - modelopt
  - dflash2
  - gb10
---

# Qwen3.8-27B custom NVFP4 local-Hessian v2 packed

This is the **packed target checkpoint** derived from
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The calibrated
ModelOpt 0.44.0 export was repaired without repeating calibration: 208
declared FP8 weights and 192 declared NVFP4 weights were packed using the
scales already present in the calibrated export; the packed U8 `lm_head`
was preserved. The source and this derivative use Apache-2.0. The loadable
model files are at the repository root; the top-level `README.md` is this card.

The [Fucina receipt](https://github.com/PRJXAI/fucina/blob/evidence/qwen38-27b-nvfp4-v2-packed-receipt/docs/evidence/qwen38-27b-custom-nvfp4-local-hessian-v2-packed/repack-execute-01.json)
binds the exact 22-file output, six safetensors shards, 2,194 tensors,
21,921,428,072 tensor payload bytes, and layout fingerprint
`12e55f382b6ea1fd04d85449e598cc8d2009a88f64788c457277ae5fce911f4b`.
Its SHA-256 is `409b7115b2f5721e741fca7bca3756351c9674456af35095ac8cd7fea131e280`.
The completed repack receipt is *not* a quality or performance admission.
The calibrated raw export needed to rerun the no-calibration repack is **not**
included here. This repository is for loading the finished packed model.
The [Fucina reproduction handoff](https://github.com/PRJXAI/fucina/blob/evidence/qwen38-27b-nvfp4-v2-packed-receipt/reproduction/README.md)
also documents an optional script for a **new** BF16-to-ModelOpt calibration
run. That script requires the pinned 512-row calibration inputs from the owner
and does not recreate this exact packed checkpoint by itself.

## Reproduce the measured SGLang configuration

The approximately **77 committed decode tokens/s** observation was for the
*target plus a separate NVFP4 DFlash2 draft* on one NVIDIA GB10 (arm64),
stock SGLang v0.5.20, one running request, and 256 committed output tokens.
The pinned draft is
[`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal)
at revision `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1`.
Downloading this target alone does **not** reproduce that throughput.

Install the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
and download the target and the pinned draft into separate local directories.
For a reproducible deployment, substitute an immutable target Hub commit for
`main` after publication:

```bash
hf download mauromedda/fucina-qwen38-v2 \
  --revision main --local-dir /models/qwen38-target-repo
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
  --revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
  --local-dir /models/qwen38-dflash2-draft
```

On a GB10 with sufficient free unified memory, no competing model job, Docker
with NVIDIA GPU access, and a free loopback port 8000, the measured container
configuration was:

```bash
docker run --rm --gpus all --network host --shm-size 16g \
  --mount type=bind,src=/models/qwen38-target-repo,dst=/models/target,readonly \
  --mount type=bind,src=/models/qwen38-dflash2-draft,dst=/models/draft,readonly \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  lmsysorg/sglang@sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862 \
  python3 -m sglang.launch_server \
  --model-path /models/target --load-format safetensors \
  --quantization modelopt --model-impl sglang \
  --served-model-name qwen38-27b-custom-nvfp4-local-hessian-v2-packed \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-cache-report --host 127.0.0.1 --port 8000 \
  --tp-size 1 --pp-size 1 --device cuda --context-length 131072 \
  --mem-fraction-static 0.72 --max-running-requests 1 \
  --max-total-tokens 150000 --chunked-prefill-size 8192 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer --linear-attn-backend triton \
  --speculative-algorithm DFLASH --speculative-draft-model-path /models/draft \
  --speculative-draft-load-format safetensors \
  --speculative-draft-model-quantization modelopt_fp4 \
  --speculative-num-steps 1 --speculative-eagle-topk 1 \
  --speculative-dflash-block-size 8 --speculative-num-draft-tokens 8 \
  --speculative-draft-window-size 2048 \
  --speculative-draft-attention-backend trtllm_mha \
  --speculative-draft-kv-cache-dtype fp8_e4m3
```

The image is the official Linux/arm64 SGLang v0.5.20 manifest at source commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`. Do **not** pass SGLang's
`--enable-unified-memory`: it is a separate hybrid-KV option, not the GB10
hardware property, and was not used in this DFlash run. Check
`curl -fsS http://127.0.0.1:8000/health` after loading. The command binds
only to loopback; expose it externally only after separate security review.

## What was actually measured

The decode benchmark committed exactly 256 tokens per request. On synthetic
repeated-token fixtures, the server reported steady 8/8 DFlash acceptance.
The decode-only results were:

| Prompt tokens | Committed decode tokens/s |
| ---: | ---: |
| 2,048 | 77.42 |
| 8,192 | 76.88 |
| 32,768 | 74.28 |
| 65,536 | 71.17 |

These figures are **not** target-only decode speed, a universal speed
guarantee, or proof of real-prompt acceptance. The same run measured cold
prefill below its supplied comparison target at all four context lengths.
The matched `tool-eval-bench` 2.0.4 69-case result was **86/100**, not
91/100 or 96/100. See the
[baseline report](https://github.com/PRJXAI/fucina/blob/evidence/qwen38-27b-nvfp4-v2-packed-receipt/reproduction/evidence/SGLANG-BASELINE-20260922.md)
for request settings, protocol and limitations. Use the same tokenizer,
draft, SGLang digest, prompt fixture, and 256-token decode definition before
comparing throughput.

The checkpoint has not been promoted for general production or multimodal
quality. Do not interpret the 77 tok/s synthetic decode result as end-to-end
throughput or as a claim that Fucina exceeds SGLang.
