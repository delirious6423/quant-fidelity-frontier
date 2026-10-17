---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
tags:
- archival-mirror
  - nvfp4
  - compressed-tensors
  - mtp
  - speculative-decoding
  - blackwell
  - qwen3_5
pipeline_tag: text-generation
---

# Archival mirror of [`sakamakismile/Qwen3.8-27B-MTP-NVFP4`](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) at revision `6d98dc1f`

> [!IMPORTANT]
> **This is not our work and it is not a new quantisation.** It is a byte-identical archival mirror
> of [`sakamakismile/Qwen3.8-27B-MTP-NVFP4`](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) — an NVFP4 quantisation carrying MTP weights — pinned at revision
> [`6d98dc1f1d5259c9582794014b73852baf20f805`](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4/tree/6d98dc1f1d5259c9582794014b73852baf20f805).
> Nothing was requantised, re-exported, converted or edited. **Use the upstream repository for
> anything current**; this exists only so that a published measurement of ours stays checkable.
>
> **Why it exists.** We compared this artifact against our own builds and published the numbers.
> A Hub revision can stop resolving — history gets squashed, repositories get renamed or removed —
> and when that happens a published comparison becomes unverifiable. Pinning the exact bytes we
> measured keeps the claim falsifiable by anyone, including against us.
>
> **Verification.** every file re-verified against the remote copy: LFS files by LFS oid sha256, small files by git blob sha1, both computed from local bytes
>
> **Credit for the artifact below belongs to its original author.** The README that follows is
> theirs, reproduced verbatim; only this banner was added.

---

# Qwen3.8-27B-MTP-NVFP4

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense 27.8B hybrid DeltaNet + full-attention, Qwen3-VL vision, native MTP head) quantized to **NVFP4 (W4A4, group 16)** with llm-compressor. **55.6 GB bf16 → 20.6 GB.**

- **MTP draft head preserved in bf16** and wired into the index — speculative decoding works out of the box (`method: qwen3_5_mtp`).
- Vision tower, `lm_head`, DeltaNet `conv1d`, and the MTP head are kept in bf16; everything else is NVFP4 W4A4.
- Quantized **GPU-resident**: the full bf16 model dispatched across 8× RTX PRO 2000 Blackwell (16 GB) with `device_map=auto` — calibration itself took **68 seconds** (32 samples × 8192 tok).
- Serves on SM120 (Blackwell) with vanilla vLLM v0.22.0 — compressed-tensors auto-detected, no `--quantization` flag.

## Measured throughput (TP=4, 128k ctx, KV fp8, RTX PRO 2000 Blackwell ×4)

| concurrency | no MTP (aggr t/s) | MTP n=3 (aggr t/s) |
|---|---|---|
| 1 | 49.0 | **72.6** |
| 2 | 93.2 | 110.9 |
| 4 | 180.9 | 232.8 |
| 8 | 318.3 | **386.9** |

Single-stream prefill (prefix cache disabled, mean of 3): **3,820 tok/s @ 8k**, 3,299 @ 32k, 2,363 @ 100k prompt tokens. GPU KV cache at TP=4/128k: 1,025,977 tokens (7.83× concurrency).

## Serve (vLLM v0.22.0)

```bash
vllm serve sakamakismile/Qwen3.8-27B-MTP-NVFP4 \
  --trust-remote-code --tensor-parallel-size 4 \
  --max-model-len 131072 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

On boards without P2P add `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`.

## ⚠️ Gotchas

- **The 15 `mtp.*` modules are listed in `quantization_config.ignore` — do not remove them.** If vLLM treats the bf16 MTP head as NVFP4 the draft breaks silently: 0% acceptance and *slower* than no MTP.
- W4A16 (NVFP4A16) does **not** serve on this architecture in vLLM 0.22 (`gptq_marlin_repack: size_n=24 not divisible by tile_n_size=64`). W4A4 only.
- This is a reasoning model — give it `max_tokens ≥ 4096` or the `<think>` phase eats the budget.
- Text-only serving shown above (`--limit-mm-per-prompt '{"image":0,"video":0}'` to be explicit); the vision tower is shipped in bf16 but multimodal serving was not benchmarked here.

## Recipe

llm-compressor NVFP4 (W4A4, group 16), `targets: [Linear]`, `ignore: [lm_head, re:.*visual.*, re:.*conv1d.*, re:.*mtp.*]`, 32 calibration samples × 8192 tok from `neuralmagic/calibration`. MTP tensors grafted back in bf16 post-save and appended to `quantization_config.ignore`.

🙏 Qwen team for the model, vLLM & llm-compressor teams for the tooling.
