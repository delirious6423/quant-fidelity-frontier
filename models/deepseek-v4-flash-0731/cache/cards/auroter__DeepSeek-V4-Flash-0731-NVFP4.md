---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - deepseek_v4
  - nvfp4
  - fp4
  - quantized
  - vllm
---

# DeepSeek-V4-Flash-0731-NVFP4

NVFP4 routed-expert quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), produced with NVIDIA's own published [Model-Optimizer `deepseek_v4` flow](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/deepseek) (modelopt 0.45.0). At the time of publication no public NVFP4 of the 0731 release existed; NVIDIA's official [DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) covers the earlier preview checkpoint only. (Upload-time chunk deduplication suggests an identical conversion already existed privately on the Hub — the cast is deterministic, so any faithful run of NVIDIA's flow yields these exact bytes.)

## Why this exists

DeepSeek-V4-Flash ships its routed experts natively in **MXFP4**, which vLLM serves through the `deep_gemm_mega_moe` kernel path. NVFP4 experts instead run on standard Blackwell FP4 tensor-core paths (vLLM's default MoE backend), opening the checkpoint to hardware outside that kernel's reach — including workstation Blackwell (SM120, e.g. RTX PRO 6000). This is a **kernel-path port, not a compression**: source and output are both ~160 GB, and the expert weights are converted **bit-exactly** (see below). What the conversion adds is a set of calibrated activation scales enabling NVFP4 W4A4 execution.

## What was quantized

| Component | Treatment |
|---|---|
| Routed experts (43 layers × 256 experts × w1/w2/w3 = 33,024 tensors) | NVFP4 W4A4: lossless weight cast + calibrated `input_scale` |
| Attention, router gate, shared experts, embeddings, LM head | Byte-identical to source (native FP8/BF16) |
| DSpark speculative-decoding module (V4's successor to MTP) | Byte-identical to source |

## The lossless weight cast

The expert weights were **not re-quantized**. MXFP4's per-32-element power-of-two (E8M0) scales are exactly representable in NVFP4's two-level scaling: the per-tensor `weight_scale_2` is pinned to `2^(k_max−8)` and each per-16 E4M3 block scale to `2^(k_j−m)`, so every NVFP4 nibble equals its source MXFP4 nibble and the reconstructed values are identical. `k_max` is shared across each expert's w1/w3 pair (fused GEMM1), matching vLLM's fused-scale expectation. The flow's accounting over this checkpoint: **8,657,043,456 / 8,657,043,456 blocks (100.0000%) bit-exact** — not one out-of-range block in the model. Weight-level quantization error versus the released checkpoint is exactly zero; the only new information is the activation scales.

## Calibration

Activation `input_scale` values (per-tensor amax over routed-expert inputs) were calibrated with the recipe defaults: 128 samples × 512 tokens, batch 4, model-parallel 4 — 64 samples each from `cnn_dailymail` and `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2`. NVIDIA's recipe pairs `cnn_dailymail` with `nvidia/Nemotron-Post-Training-Dataset-v2`, which is gated; the substitute is an ungated dataset from the same NVIDIA post-training family and modality (multi-turn instruction conversations). 36 of 66,048 quantizer amax entries belonged to experts that received no calibration tokens (predominantly the deterministic hash-routed early layers) and use the recipe's documented per-projection fallback.

## Verification

All gates run against the finished artifact on disk:

| Gate | Result |
|---|---|
| Key census: source − 33,024 dropped `.scale` + 99,072 added scale tensors = 138,365 keys | exact |
| Expert tensor forms (packed shape/dtype, scale shapes) across all 33,024 experts | 0 defects |
| Scale sanity full sweep (NaN/zero weight scales, power-of-two `weight_scale_2`, positive finite `input_scale`) | 0 defects |
| Independent bit-equality audit, 387 sampled tensors across all 43 layers: dequantized values vs source | exact (the only encoding difference anywhere is E2M1 −0 → +0 canonicalization, numerically indistinguishable) |
| Passthrough byte-equality (attention, shared experts, DSpark/MTP, embeddings, head) | identical |
| Config / index / manifest integrity | clean |

## Performance

Both arms measured on the identical deployment below (stock vLLM 0.26.0, TP4 + expert parallelism, marlin MoE backend, FP8 KV cache) — the native checkpoint served as the reference under the same recipe on the same hardware:

| Measurement | NVFP4 · marlin (W4A16) | NVFP4 · native W4A4 (cutlass) | Source MXFP4 · marlin |
|---|---|---|---|
| Teacher-forced perplexity — fixed WikiText corpus, 48 chunks, 65,561 scored tokens | **5.160 / 5.182** (two runs) | 5.297 | 5.178 / 5.189 (two runs) |
| Single-stream decode | 119.5 tok/s | 124.5 tok/s | 118.3 tok/s |
| 8 concurrent streams, aggregate | 661 tok/s | 701 tok/s | 670 tok/s |
| Long-form output hygiene — 50,700 completion tokens across 39 generations | 0 stray-script codepoints, 0 U+FFFD | 0 stray-script codepoints, 0 U+FFFD | — |

The two marlin columns are quality parity: that backend dequantizes both formats to BF16, the weight cast is bit-exact, and the same-arm run-to-run spread (up to 0.022, from batching nondeterminism) is as large as any difference between them. The native W4A4 column is the FP4 tensor-core path this artifact exists for, and it shows the expected trade: activations quantized to FP4 buy +4–6% throughput on SM120's current kernel stack at a ~2.5% perplexity cost. Prefill speed was unchanged between backends on this stack. Larger W4A4 gains should arrive as the SM120 cutlass MoE path matures; the quality trade is intrinsic to W4A4.

### With DSpark speculative decoding (out-of-tree SM120 branch)

Same hardware, vLLM built from [jasl/vllm](https://github.com/jasl/vllm) tag `sm120-pr-41834-stable-preview-20260727d` (tracking [vLLM PR #41834](https://github.com/vllm-project/vllm/pull/41834)) plus the one-file draft-routing patch shipped in this repo:

| Configuration | Single-stream decode | 8-stream aggregate | Perplexity (same corpus) |
|---|---|---|---|
| NVFP4 + DSpark | 168.7 tok/s | 659 tok/s | 5.294 |
| NVFP4 + DSpark, V2 runner (`VLLM_USE_V2_MODEL_RUNNER=1`) | **208.4 tok/s** | 663 tok/s | — |
| Native MXFP4 + DSpark, V2 runner (reference) | 210.5 tok/s | 637 tok/s | 5.169 |

Speculation is lossless (perplexity stays at each backend's baseline) and lifts single-stream decode ~67% over the best non-speculative configuration; at 8 concurrent streams it is throughput-neutral. To our knowledge this is the first DSpark speculation running on an NVFP4-form checkpoint.

**The draft-routing patch is required.** NVFP4 conversions of this model quantize only the main 43-layer stack; the DSpark draft module keeps its original MXFP4 experts (byte-identical here — see the table above). The branch's loader routes all routed-experts layers through the NVFP4 path when the checkpoint declares NVFP4, which decodes the draft's MXFP4 weights as noise: the engine runs but draft acceptance is 0% and decode collapses below the non-speculative rate. [`dspark-nvfp4-draft-routing.patch`](./dspark-nvfp4-draft-routing.patch) (43 lines, Python-only, no recompile) makes the routing prefix-aware so draft-module layers (runtime index ≥ `num_hidden_layers`) load through the MXFP4 path. With it, draft acceptance reaches ~41% on the V2 runner.

All throughput figures are **non-speculative**: the checkpoint's DSpark speculative-decoding module has no serving path on stock SM120 wheels (see Deployment), so both arms were measured with it off. These are floor numbers — deployments where DSpark runs (datacenter Blackwell, or the out-of-tree SM120 branch) will decode faster.

## Deployment

Verified with stock vLLM 0.26.0 wheels on 4× RTX PRO 6000 Blackwell Workstation (SM120) — to our knowledge the first NVFP4 serving of this model on workstation Blackwell:

```bash
# nvcc must be on PATH (TileLang and DeepGEMM JIT-compile kernels at first boot);
# a CUDA 13.0 toolchain matching torch's cu130 runtime avoids header-guard failures.
export NCCL_P2P_DISABLE=1        # Blackwell PCIe P2P allreduce deadlocks

vllm serve <path> \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --kernel-config '{"moe_backend":"marlin"}' \
  --trust-remote-code
```

The native W4A4 path is also verified on SM120, with the trade shown in the Performance table (+4–6% throughput, ~2.5% perplexity): use `--kernel-config '{"moe_backend":"flashinfer_cutlass"}'` and **omit** `--enable-expert-parallel` — the cutlass MoE path has an open NaN hazard specifically under expert parallelism (vLLM issue #42118); with plain TP4 we measured clean output over the full gate suite.

SM120 notes: leave DeepGEMM enabled (`VLLM_USE_DEEP_GEMM=0` routes this checkpoint's UE8M0 scales into kernels that cannot read them — vLLM PR #47988 carries the real fix); single-request context tops out at 256K on stock wheels.

For DSpark speculative decoding (the fastest configuration measured above), build [jasl/vllm](https://github.com/jasl/vllm) at tag `sm120-pr-41834-stable-preview-20260727d` from source (no wheels exist; torch 2.13.0+cu130, FlashInfer 0.6.15.post1, single-arch `TORCH_CUDA_ARCH_LIST=12.0a`), apply the [draft-routing patch](./dspark-nvfp4-draft-routing.patch), and serve with `--tokenizer-mode deepseek_v4 --speculative-config '{"method":"dspark","num_speculative_tokens":5}'` (5 is the checkpoint's `dspark_block_size`), `VLLM_USE_V2_MODEL_RUNNER=1` for the higher-acceptance speculator, and expert parallelism off. On datacenter Blackwell (B200/B300) the standard NVFP4 deployment from NVIDIA's preview card applies unchanged.

## Hardware

Built and tested on a single workstation: 4× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120), 188 GB system RAM. Build cost end to end: model-parallel conversion 3m40s (CPU), calibration 4m08s on the four GPUs, export ~25 min (CPU, I/O-bound).

## Licensing

MIT, inherited from [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (LICENSE included). The license governs the weights; the routed-expert tensors are a bit-exact re-encoding of the source weights plus derived scale metadata, and all other tensors are byte-identical copies.
