---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- qwen3.8
- autoround
- int4
- w4a16
- vllm
- mtp
- speculative-decoding
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-int4-AutoRound

INT4 (W4A16) AutoRound quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — **18 GB**, with the **MTP speculative-decoding head quantized and working**.

The MTP head is quantized along with the model, so `--speculative-config` in vLLM works out of the box and roughly **doubles decode throughput** (measured +98 % on temp-0 prose vs the same GPU without MTP; draft acceptance 41–67 % depending on content).

## Why another int4?

At the time of upload, the available 4-bit options either dropped the MTP tensors
(no speculative decoding) or kept hundreds of tensors in bf16 (28 GB — no longer
fits a 24–32 GB GPU next to a useful KV cache). This one follows the recipe that
served Qwen3.6-27B in production for months:

- **W4A16, group size 128, symmetric**, `auto_round:auto_gptq` packing (AutoRound 0.14.2, default tuning)
- Quantized blocks: `model.language_model.layers` **and `mtp.layers`**
- The 96 GDN linear-attention `in_proj_a` / `in_proj_b` tensors are kept **bf16** (quantizing them hurts the hybrid-attention path far more than it saves)
- Vision tower and embeddings untouched (bf16) — multimodal input still works

## Measured (vLLM 0.22/0.23)

| Setup | Result |
|---|---|
| Weights in VRAM | ~18.0 GiB (TP1) / 9.1 GiB per rank (TP2) |
| RTX 5090 @ **400 W cap**, TP1, fp8 KV, 215K ctx | prefill ~2,580 tok/s (74K cold doc), decode **124 tok/s** with MTP n=3 / 59 without |
| 2× RTX 3090 @ **200 W cap each**, TP2, fp8 KV, 262K ctx | 2.22× concurrency at full 262,144 ctx; decode ~73 tok/s with MTP |
| MTP draft acceptance | 41–46 % (prose) · ~67 % (mixed) · higher on code/predictable text |
| Tool calling | works with `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` |

## Serving example

```bash
vllm serve Frozenlock/Qwen3.8-27B-int4-AutoRound \
  --quantization auto_round \
  --max-model-len 215000 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Context is 262,144 native (fits on 2×24 GB at TP2 with fp8 KV; on a single 32 GB
card ~215K leaves KV headroom). The base model supports YaRN extension to 1M —
see the [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B) for the
`rope_parameters` block; budget ~35 KB of KV per token (fp8) when sizing.

`reasoning_effort` (`none`/`low`/`medium`/`xhigh`) works as in the base model.
Note that sending `chat_template_kwargs {"enable_thinking": false}` makes the
template silently ignore `reasoning_effort` — use one knob or the other.

## Notes

- All speeds above were measured on **power-capped** GPUs (5090 at 400 W of a 600 W stock limit; 3090s at 200 W of 350 W stock). Stock power limits should do the same or better — the 3090 decode figures in particular leave headroom.
- Calibration: AutoRound default dataset/iters, batch 4 × grad-accum 2, on 4× RTX 3090.
- On Ampere/Blackwell the weights run through the Marlin W4A16 kernels; prefill speed is unchanged vs 8-bit (compute-bound), decode gains come from smaller weight reads, and the big win vs FP8/NVFP4 is the KV-cache room the smaller weights leave.
- Quantized with [intel/auto-round](https://github.com/intel/auto-round) 0.14.2.
