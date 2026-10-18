---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: transformers
tags: [auto-round, w4a16, int4, gptq, moe, deepseek]
---

# DeepSeek-V4.1-Flash — W4A16 (experts INT4 GPTQ + Engram INT4) — AutoRound

Post-training quantization of [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
produced with [auto-round](https://github.com/intel/auto-round) `--model_free` (RTN, iters=0):

| Component | Official | This checkpoint |
|---|---|---|
| Routed experts (`ffn.experts.*`, incl. MTP) | MXFP4 (packed E2M1 + E8M0/32) | **INT4 W4A16 g32 sym, auto_gptq packing** (`qweight`/`qzeros`/`scales`) |
| Engram embedding tables (`layers.1/14.engram.embed`) | FP8 (E4M3 + E8M0/32) | **INT4 g32 sym** (packed uint8 `[R,128]` + fp16 scale `[R,8]`) |
| Attention / shared_experts / engram.wkv | FP8 / MXFP8 | BF16 (dequantized) |
| Vision tower / aligner / embed / head / norms | BF16 | BF16 (unchanged) |

Checkpoint size: **451.7 GB** (vs 510.3 GB official). gsm8k (raw 5-shot, thinking off, n=1319):
**93.93 / 94.01** vs baseline 92.87 — lossless within noise.

## Inference (vLLM)

Requires: vLLM `main` (≥ PR [#56201](https://github.com/vllm-project/vllm/pull/56201))

```bash
DSV41_ENGRAM_DTYPE=int4 NCCL_NVLS_ENABLE=0 \
vllm serve <this-model-dir> \
  --tensor-parallel-size 4 --max-model-len 8192 --language-model-only \
  --gpu-memory-utilization 0.90 --port 8100
```

Verified on 4×H200 (TP4). The INT4 expert GEMM runs through vLLM's built-in GPTQ-MoE
path (MoeWNA16/Marlin) — no custom kernels; dense layers run as plain BF16.

## Quantization recipe

```bash
# experts only -> auto_gptq INT4 g32 (everything else ignored / dequantized to bf16)
auto-round --model_name deepseek-ai/DeepSeek-V4.1-Flash --model_free \
  --scheme W4A16 --group_size 32 \
  --ignore_layers attn,shared_experts,engram,vision,aligner,main_proj,gate,norm,embed,head \
  --output_dir <A>

# engram tables -> int4 g32 sym + fp16 scale (custom RTN script), then splice into A
# (see producer notes: quant_engram_int4.py + build_b_view_experts.py)
```

## Limitations

- Serving today requires the `dsv41-quant-plugin` (not yet upstreamed to vLLM) and
  `DSV41_ENGRAM_DTYPE=int4` for the engram lookup.
- Not compatible with the official reference implementation's FP8 kernels (dense is BF16 here).

## Provenance

- auto-round 0.15.0 (editable copy with DeepSeek-V4.1 source-format patch), vLLM main nightly
  (`0.1.1.dev39+g46d2b23ac`), transformers-side loading not used (model_free).
- Full production log & design docs: see producer's journal (aqa pipeline).
