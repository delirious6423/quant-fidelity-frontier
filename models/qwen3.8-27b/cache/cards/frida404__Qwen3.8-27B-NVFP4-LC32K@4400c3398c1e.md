---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - fp8
  - modelopt
  - qwen3.8
  - speculative-decoding
  - long-context
---

# Qwen3.8-27B-NVFP4-LC32K (a2gdn-lc32k-mtpa4)

A post-training-quantized `Qwen/Qwen3.8-27B` for NVIDIA Blackwell GPUs, made to serve on a single 96 GB card. It puts
more of the model in NVFP4 than `nvidia/Qwen3.8-27B-NVFP4` (the 48 GDN linear-attention projections and the MTP head),
and recovers the long-context accuracy that this normally costs by calibrating on 32K-token documents.

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) by the Qwen team,
  Alibaba Cloud. All weights and scales are computed from the BF16 base. The recipe uses the published
  `nvidia/Qwen3.8-27B-NVFP4` only as a scaffold for the MTP graft; the MTP tensors it carries over unquantized
  (`mtp.fc` and the MTP norms) are bitwise identical to the BF16 base's, and no quantized weight or scale of the
  published checkpoint is used.
- **Quantization:** NVIDIA Model Optimizer (ModelOpt) PTQ, layerwise Local-Hessian weight calibration.
- **Not affiliated with or endorsed by** the Qwen team or NVIDIA.
- **Code, patches and reproduction:** [Qwen3.8-27B-NVFP4-SingleGPU-Serving](https://github.com/Frida-a/Qwen3.8-27B-NVFP4-SingleGPU-Serving).

## What is quantized

| module | format | count |
|---|---|---|
| MLP gate / up / down | NVFP4 W4A4 (static weights, dynamic activations, 16-element blocks) | 64 layers |
| GDN in_proj_qkv / in_proj_z / out_proj | **NVFP4 W4A4** (FP8 in the published checkpoint) | 48 layers |
| full attention q / k / v / o | FP8 W8A8, per-tensor | 16 layers |
| lm_head | NVFP4 | 1 |
| MTP head linears (q/k/v/o, gate/up/down) | **NVFP4** (BF16 in the published checkpoint) | 1 block |
| embeddings, GDN in_proj_a/b, conv1d, norms, vision tower | BF16 | |

18.96 GB of safetensors (18.65 GB of trunk shards + a 0.31 GB NVFP4 MTP shard; published NVFP4 checkpoint: 21.92 GB,
so -13.5%). No KV-cache scales are stored; serve with an FP8 (or K8V4) KV cache.

## Recipe

1. Long-context calibration set: 128 records x 32,768 tokens (4.19M tokens) = 32 PG-19 book windows, 32 code-repository
   concatenations (The Stack v1 via ProLong), 16 textbook chapters (ProLong), 32 synthetic multi-turn recall chats
   (Nemotron SFT instruction-following chat turns as the haystack, PG-19 sentences as needles) and 16 OpenHands agent
   trajectories (SWE-rebench). Every record was checked with a 13-gram / 50-gram overlap rule against GPQA-D,
   IFBench, SciCode, LiveCodeBench v6, MRCR-128K, an MMMU-Pro subset, an MMLU-Pro subset and AA-LCR (HLE was not
   indexed); long-context benchmark corpora were never used as sources.
2. ModelOpt `hf_ptq` with layerwise Local-Hessian calibration, FP8 scale sweep and shared global amax for fused
   projections; lm_head quantizer off.
3. NVFP4 lm_head taken from a reproduction of the published NVFP4 recipe (2,048 x 2,048 tokens of Nemotron
   post-training v3).
4. MTP head quantized separately on BF16 hidden states (128 x 2,048 tokens of Nemotron post-training v3; top-1
   agreement with the BF16 head 0.958) and grafted.

Calibration length is the point: the same layout calibrated on 2,048-token samples lost 2.30 points of MRCR-128K; a
62%-long / 38%-short mix lost the benefit entirely.

## Evaluation

NeMo Gym v0.7.0, model-card sampling (temperature 1.0, top_p 0.95, thinking on, 65,536 max output tokens), FP8 KV
cache, no speculative decoding, on GB300. Deltas are vs pooled runs of `nvidia/Qwen3.8-27B-NVFP4` on the same harness,
+- 1 standard error.

| benchmark | nvidia/Qwen3.8-27B-NVFP4 (ours, pooled) | this model | delta |
|---|---|---|---|
| GPQA-Diamond (x4) | 88.21 | 87.88 | -0.33 +- 0.67 |
| IFBench, stripped / raw (x2) | 80.81 / 74.81 | 82.67 / 76.50 | +1.9 / +1.7 |
| MRCR-128K (1,492 conversations, 2 runs) | 94.01 | 92.93 | **-1.10 +- 0.50** |
| HLE (hle_verified) | 41.46 (2 runs) | 40.62 | -0.84 +- 0.92 |

- The MRCR-128K loss is real (95% CI -2.08 to -0.11) and inside the -1.5 accuracy limit, with no
  training. The published checkpoint itself is ~1.3 below BF16 on the same benchmark.
- Output length and truncation are unchanged on GPQA. Premature end-of-sequence inside `<think>` is more
  frequent on HLE only, where it replaces would-be 64K truncations (no-answer rate unchanged: 28.1% vs 28.0%).
- Single-run numbers carry about +-1 point of job-to-job noise beyond the quoted SE.

## Serving

Built and measured with vLLM nightly `f5d93296` plus the patch series in [Qwen3.8-27B-NVFP4-SingleGPU-Serving](https://github.com/Frida-a/Qwen3.8-27B-NVFP4-SingleGPU-Serving). Upstream vLLM at that
commit loads the trunk but **not** the NVFP4 MTP head (the MTP drafter drops per-layer quant entries; patch group 03
fixes it).

Default (throughput) profile on one RTX PRO 6000 Blackwell (96 GB):

```bash
VLLM_KV_K8V4=1 VLLM_GDN_SPEC_STATE=recover VLLM_GDN_RECOVER_FOLD=1 \
vllm serve <this repo> --served-model-name qw38 --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 131072 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.85 --max-num-seqs 256 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --async-scheduling --mamba-ssm-cache-dtype int16 --block-size 1600 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","num_speculative_tokens_per_batch_size":[[1,32,3],[33,1024,1]]}'
```

The environment variables and `--mamba-ssm-cache-dtype int16` need the patched build. For a low-latency profile
(<= 16 concurrent requests, DFlash2 NVFP4 drafter) see [docs/FINAL_CONFIG.md](https://github.com/Frida-a/Qwen3.8-27B-NVFP4-SingleGPU-Serving/blob/main/docs/FINAL_CONFIG.md).

Recommended sampling (from the base model): temperature 1.0, top_p 0.95, top_k 20; thinking on by default.

Speed on one RTX PRO 6000 Blackwell Max-Q at 250 W, measured in one session against the published
`nvidia/Qwen3.8-27B-NVFP4` served with its model-card recipe plus MTP k=3 on stock vLLM:

| | published NVFP4, card recipe + MTP | this model, default profile | this model, latency profile |
|---|---|---|---|
| real prompts, c=1, tok/s/user | 116.6 | 162.3 (+39%) | **216.9 (+86%)** |
| 1K/1K synthetic, c=64 / c=128, tok/s/GPU | 1,651 / 1,602 | **2,586 / 2,848** (+57% / +78%) | |
| 32K/1K synthetic, c=16, tok/s/GPU | 149 | 227 (+52%) | |
| TTFT, 124K-token prompt | 36.6 s | 19.3 s | |
| KV-cache capacity (tokens) | 1.25M | 2.01M | 1.10M |

These numbers combine the checkpoint with the patched serving stack. On the same patched stack, this checkpoint alone
is +8.8 / +8.4 / +12.1% tok/s/GPU at c=1/4/16 over the published checkpoint, with unchanged MTP acceptance.

**Served-stack accuracy:** both serving profiles passed a final GB300 gate against the previous build of the stack
(MRCR-128K, MRCR >128K, GPQA-D, HLE, IFBench). The largest negative deltas are both in the latency
profile and within noise: IFBench stripped -1.67 (one run, inside the +-2.4 single-run spread) and HLE -0.67 +- 0.46.
See docs/RESULTS.md 3.2 in the repository.

## Limitations

- Long-context retrieval is about 1 point lower than `nvidia/Qwen3.8-27B-NVFP4` on MRCR-128K.
- Accuracy is reported on English reasoning (GPQA-Diamond, HLE), instruction following (IFBench) and long-context
  retrieval (MRCR) benchmarks only; no safety, multilingual, agentic or tool-use accuracy evaluation was run. Behaviour on those follows
  the base model at best and may be degraded by quantization.
- No accuracy on vision inputs is reported.
- Quantized for NVIDIA Blackwell (NVFP4 tensor cores); other GPUs will not run the NVFP4 GEMMs natively.
- Inherits the base model's limitations and biases; see the Qwen3.8-27B model card.

## License and attribution

Apache-2.0, as the base model. This is a quantized derivative of Qwen/Qwen3.8-27B (Copyright Alibaba Cloud,
Apache-2.0); the base model's LICENSE file is included unchanged and its notices apply. Quantized with NVIDIA Model
Optimizer (Apache-2.0). Calibration data was used only to compute quantization scales and is not redistributed:
PG-19 (Apache-2.0), ProLong long-context data (The Stack v1 terms; TextbookChapters), Nemotron-SFT-Instruction-
Following-Chat-v2 (ODC-By), SWE-rebench OpenHands trajectories (CC-BY-4.0), Nemotron post-training dataset v3.

If you use this model, please cite the Qwen3.8 base model as its card asks.
