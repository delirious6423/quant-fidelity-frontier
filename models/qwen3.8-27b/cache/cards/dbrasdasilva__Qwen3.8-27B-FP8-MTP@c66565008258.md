---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - qwen3.8
  - fp8
  - quantized
  - modelopt
  - mtp
  - speculative-decoding
  - blackwell
  - vision
  - multimodal
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

# Qwen3.8-27B-FP8-MTP — vision tower AND MTP kept

A quantized build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) that keeps
**both** the vision tower and the MTP head, exported in **`modelopt`** format for the native
path on NVIDIA Blackwell / SM120.

Quantized from the **bf16 base**, not from another quantized checkpoint.

## What is kept, and at what precision

| Component | Precision |
|---|---|
| Vision tower (333 tensors) | **BF16 — never quantized** |
| MTP head (15 tensors) | **BF16** — grafted from the base after export, so speculative decoding works |
| `lm_head` | **BF16** — see the note below, this is a requirement |
| Attention projections | FP8 E4M3 |
| `linear_attn` norms, `in_proj_a/b`, `conv1d` | BF16 |
| MLP | **FP8 E4M3** for all 64 layers |
| KV cache | FP8 E4M3 with **calibrated `k_scale`/`v_scale`** (16 each) |

Size on disk: **30 GB**.

## Why `lm_head` is left unquantized

Not because vLLM cannot load a quantized one — `nvidia/Qwen3.8-27B-NVFP4` ships an
NVFP4 `lm_head` and loads fine. It is left in BF16 here for two reasons: it is a small
fraction of the weights, and it keeps `quantization_config.ignore` honest.

If you quantize this architecture yourself, the thing that actually bites is a
**mismatch between the config and the weights**. An earlier build of this model listed
`lm_head` in `quantization_config.ignore` while having quantized it, and vLLM refused
to load it:

```
ValueError: There is no module or parameter named 'lm_head.input_scale' in Qwen3_5ForCausalLM.
The available parameters belonging to lm_head (ParallelLMHead) are: {'lm_head.weight'}
```

vLLM builds the module from the config, then finds scale tensors the config said would
not be there. Either quantize it and say so, or do not quantize it and say that —
just do not ship both claims.

## Calibrated KV-cache scales

This checkpoint carries `k_scale`/`v_scale` for the 16 layers that use full attention
(the other 48 are gated-delta linear attention and have no KV cache). Without them,
serving with `--kv-cache-dtype fp8` silently runs at scale 1.0:

```
Using KV cache scaling factor 1.0 for fp8_e4m3
Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention
```

`q_scale`/`prob_scale` are still absent — modelopt's KV quantizer covers K and V only,
and vLLM substitutes `k_scale` for q.

## Serving

Verified on **vLLM 0.30.0** (official image) on an RTX PRO 6000 Blackwell 96 GB,
driver 615.71.09. Note there is **no** `--language-model-only`:

```bash
vllm serve Qwen3.8-27B-FP8-MTP \
  --trust-remote-code --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-model-len 262144 --max-num-seqs 4 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 \
  --quantization modelopt \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Measured: weights load in ~14 s, **1,622,656 KV tokens**, 6.19× concurrency at 262 k context.
It answers questions about images — the vision path was tested by inference, not only by
inspecting the files.

**Budget for reasoning tokens.** With `--reasoning-parser qwen3` the model thinks before
answering and that counts against `max_tokens`. At 64 the entire budget went to reasoning
and `content` came back `null`. Use 256+.

## How it was built

- `nvidia-modelopt` 0.43.0, torch 2.14.1+cu130, transformers 5.18.0
- Calibration: `neuralmagic/calibration` (LLM split), **64 samples**, seqlen 8192
- 2140 quantizers inserted; ~38 s of GPU time for calibration
- The MTP head is grafted back as raw bf16 tensors after export, because it is not a
  live module during calibration

Calibration is **text-only**, which every public text-calibrated VLM quantization shares.
The vision tower is unquantized so it is unaffected, but the quantized language layers
are calibrated on text distributions.

## Companion build

[`Qwen3.8-27B-NVFP4-MTP`](https://huggingface.co/dbrasdasilva/Qwen3.8-27B-NVFP4-MTP) — same base, same calibration, same everything except MLP precision,
so a comparison between the two attributes to precision alone.

## Reproducing this

Built in a disposable container on the host that owns the GPU: a CUDA image with
`nvidia-modelopt`, the driver userspace injected by `nvidia-container-toolkit` so it
matches the host exactly, and the base model plus exports on a separate volume that
outlives the container. The whole quantization run is ~90 seconds on this card; the
55.6 GB base download dominates everything.

The precision plan was read out of `unsloth/Qwen3.8-27B-NVFP4`'s shipped
`quantization_config` rather than guessed, then corrected in two places: `lm_head`
moved out of the FP8 tier (see above), and `*[kv]_bmm_quantizer` added from
`modelopt`'s own `FP8_KV_CFG` to get calibrated KV scales.
