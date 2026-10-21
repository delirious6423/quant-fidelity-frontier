---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- qwen3_5
- w4a16
- compressed-tensors
- int4
- volta
- v100
- sm70
- mtp
- speculative-decoding
- vllm
- cmp-170hx
- a100
language:
- en
pipeline_tag: text-generation
---

# Qwen3.8-27B-MTP-W4A16-VOLTA-Ampere

<p align="center">
  <img src="israeliAI.png" alt="Israeli AI" width="420"/>
</p>

INT4 (W4A16) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
built specifically so the model runs on **pre-Hopper GPUs** — including
**Volta (V100, sm_70)**, which cannot run the official FP8 release — with the
model's **native MTP speculative-decoding head fully intact and working**.

51.7 GB BF16 → **25.8 GB**. Fits and serves on 2× V100 32GB (TP2) with
256K full context, and on any Ampere+ GPU via stock vLLM compressed-tensors
kernels.

## Measured (2× V100 SXM2 32GB, TP2, fp8_e5m2 KV, CUDA graphs on)

| metric | value |
|---|---|
| decode, MTP spec-decode (3 draft tokens) | **43–68 tok/s** (53.5 typical) |
| MTP mean acceptance length | 2.2–2.8 of 4 |
| context served | 131,072 (256K native; hybrid-GDN KV is tiny) |

## What is quantized (and what deliberately is not)

Quantized to INT4, **symmetric, group_size 128**, compressed-tensors
`pack-quantized` (RTN):
- text-decoder `self_attn.{q,k,v,o}_proj` (the 16 full-attention blocks)
- text-decoder `mlp.{gate,up,down}_proj` (all 64 layers)

Kept in full precision (BF16/FP16):
- **the entire MTP head (`mtp.*`)** — this is what keeps native
  speculative decoding working. (We A/B'd against a calibrated AWQ+GPTQ
  community quant of the same model: it loads fine but its MTP acceptance
  collapses to ~1.0 — drafts always rejected — cutting speed by ~3×.
  Plain symmetric RTN with an untouched MTP head wins end-to-end.)
- all hybrid linear-attention / Gated-DeltaNet tensors
  (`linear_attn.in_proj_*`, `out_proj`, `conv1d`, `A_log`, `dt_bias`)
- the vision tower (`visual.*`), embeddings, `lm_head`, all norms

Symmetric-only on purpose: asymmetric INT4 has no kernel on the sm_70
path; symmetric group-128 runs everywhere from Volta to Blackwell.

## Serving

### Volta (V100) — tested
Stock vLLM wheels drop sm_70; use a Volta-capable vLLM build
(e.g. the 1Cat-vLLM fork). Key environment:

```bash
export VLLM_SM70_QUANT_BACKEND=turbomind
export VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve Israeli-AI/Qwen3.8-27B-MTP-W4A16-VOLTA-Ampere \
  --dtype float16 --trust-remote-code \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 --max-num-seqs 4 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_V100 \
  --compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4]}' \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3
```

The exact serve script used for the numbers above ships in this repo
(`serve-qwen38-27b.sh`), along with the quantization script
(`quant-qwen38.py`).

### Ampere and newer — should work, untested
Mainline vLLM reads compressed-tensors W4A16 natively (Marlin/CT
kernels); drop the `VLLM_SM70_*` variables and the V100 attention
backend. Reports welcome.

## Usage notes
- Qwen3.8 is a **thinking model**. For direct answers send
  `"chat_template_kwargs": {"enable_thinking": false}` in the request.
- The vision tower is preserved at full precision but multimodal serving
  was **not tested** on the Volta path; text-only is verified.
- `quantization_config.ignore` uses broad regex patterns
  (`re:.*visual.*` etc.) on purpose: vLLM matches ignore rules against
  its *internal* module paths, which drop the `model.language_model.`
  prefixes — narrow patterns cause
  `assert input_size_per_partition % group_size == 0` failures at load.

## Recipe
Data-free RTN, symmetric int4, group 128, packed with
`compressed_tensors.pack_to_int32`. 256 projections quantized, everything
else copied through. No calibration set — at this size and group width,
temp-0 outputs were indistinguishable from a calibrated AWQ+GPTQ variant
in our tests, and the untouched MTP head decisively outperforms it.

Quantized and served by the Israeli-AI homelab stack. Same lineage as
[DeepSeek-v4-flash-180b-W4A16-VOLTA-Ampere](https://huggingface.co/Israeli-AI/DeepSeek-v4-flash-180b-W4A16-VOLTA-Ampere).