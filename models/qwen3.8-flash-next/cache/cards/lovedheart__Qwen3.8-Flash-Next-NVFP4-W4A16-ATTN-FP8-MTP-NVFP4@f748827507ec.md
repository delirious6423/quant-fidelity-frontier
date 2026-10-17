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
- FP4
- fp4
- FP8
- fp8
- mixed-precision
- blockwise
- speculative-decoding
- MTP
- sglang
---

> ## ⚠️ Requires a patched SGLang — stock builds emit GARBLED OUTPUT
>
> This is a **mixed-precision** checkpoint (main model identical to
> `Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-full`): NVFP4 W4A16
> 4-over-6 routed experts, FP8 E4M3 2D-blockwise (128×128) weight-only
> attention projections (`FP8_PB_WO`), **plus a re-quantized MTP draft
> layer** (`mtp.*` keys with their own `quantized_layers` entries).
>
> **Stock SGLang does not dispatch `FP8_PB_WO`** and has no NextN-draft
> quant-config plumbing for these layouts; the server may start silently
> and produce garbage or a dead draft. Use the `rebase/qsa-on-main`
> branch of https://github.com/lovedheart/sglang (sgl-project/sglang
> based). Startup sanity check: log shows `quant=modelopt_mixed`,
> `Detected nvfp4 checkpoint`, **zero** `not found in params_dict, skip
> loading` lines for `mtp.*`. For this checkpoint the draft experts must
> resolve to **`NvFp4MoEMethod`** and the draft attention to
> **unquantized (BF16)** — if the draft shows `Fp8MoEMethod`, the v1b/v3
> tensors were loaded instead.

# Model Overview

## Description
This is a **speculative-decoding (MTP/NextN) tuned variant**
The **main (target) model is untouched**: 48 layers of NVFP4 W4A16
(4-over-6) routed experts + FP8-block-128 weight-only attention, exactly
as in the base checkpoint (https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-4-Over-6-FP8).

**The only difference is in the single MTP draft layer:**

| MTP submodule | base `-full` | this checkpoint (v4) |
|---|---|---|
| routed experts (512, top-10) | NVFP4 W4A4 | **NVFP4 W4A16, 4-over-6 scheme, group 16, activations kept FP32/BF16 (no input quant)** |
| shared expert | BF16 | BF16 (byte-identical to `-full`) |
| attention q/k/v/o + norm + router + embed | BF16 | **BF16 (byte-identical to `-full`)** |

The draft experts were re-quantized offline from the BF16 source weights
using a **4-over-6** (non-uniform FP4 grid refinement) scheme, which
halves draft-expert weight memory vs the FP8-block-128 experiment while
measurably *improving* draft acceptance (see below). Draft **attention
is deliberately kept BF16**: FP8 attention on the draft (v1/v2
checkpoints) decorrelates the draft from the target and collapses
acceptance to ~1.0.

## Files specific to this variant
- `model-mtp-nvfp4-moe.safetensors` — all 4,608 draft-expert tensors
  (per-expert `gate/up/down_proj` weight + FP8-E4M3 block scale + FP32
  `weight_scale_2`), single deduplicated copy
- `model-mtp-bf16-attn.safetensors` — draft attention BF16 weights
- all other shards are hard-linked from the base/v3 checkpoints
  (main model bytes identical); checkpoint unique footprint ≈ 122 GB

## Speculative-decoding behavior (measured, RTX PRO 6000 / SM120, tp=1)
Same launch command as the base checkpoint with NEXTN
(`--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4 --speculative-draft-kv-cache-dtype fp8_e4m3`):

| draft config | accept_length | accept_rate |
|---|---|---|
| BF16 draft (base `-full`) | 2–3 | — |
| **BF16-attn + NVFP4-4o6 experts (v4, this)** | **2.5–2.6** | **~0.53** |

## Use
```sh
python -m sglang serve \
  --model-path /media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-mtpV4 \
  --served-model-name Qwen3.8-Flash-Next --trust-remote-code \
  --tensor-parallel-size 1 --max-running-requests 4 \
  --chunked-prefill-size 4096 --max-prefill-tokens 8192 \
  --mem-fraction-static 0.9596 --kv-cache-dtype nvfp4 \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-backend flashinfer --page-size 64 \
  --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_b12x \
  --attention-backend triton --decode-attention-backend trtllm_mha \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --speculative-algo NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --ple-offload-embedding --sleep-on-idle --enable-metrics --port 8070
```

### License/Terms of Use
See the [source model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for license terms.

## Model Limitations
Same as the base checkpoint: the draft is a speed-only optimization and
inherits all target-model limitations (toxicity/bias from training
data, potential factual errors). Quantized draft experts can accept
fewer tokens on math-heavy continuations; worst case degrades to
draft-less speed, never to wrong output.
