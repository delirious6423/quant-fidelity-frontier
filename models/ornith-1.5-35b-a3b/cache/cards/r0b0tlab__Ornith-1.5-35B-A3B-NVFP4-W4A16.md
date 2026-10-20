---
license: mit
library_name: sglang
pipeline_tag: image-text-to-text
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
- nvfp4
- w4a16
- modelopt
- sglang
- blackwell
- gb10
- sm121
- mtp
- speculative-decoding
- quantized
---

# Ornith-1.5-35B-A3B-NVFP4-W4A16

NVFP4 **W4A16 weight-only** quantization of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(revision `e4dfb35a93d4b6822a811a7676f3488514abe7e2`), produced with
NVIDIA Model Optimizer (dev `913f5e224`) using the official `w4_nvfp4`
recipe preset. FP4 weights (block-16, E4M3 block scales) with BF16
activations, plus an FP8-E4M3 KV cache cast. The bundled MTP head is kept
in BF16 and served with SGLang EAGLE/nextn speculative decoding.

- **Size:** 23 GB on disk (vs 67 GB BF16 source) — 2.9× compression
- **Architecture:** Qwen3_5MoeForConditionalGeneration (35.1B params, hybrid
  GDN linear attention + softmax attention + MoE, multimodal)
- **Serving:** SGLang ≥ 0.5.6.post3 (`5a7b26c63`) on Blackwell-class NVIDIA
  GPUs; validated on DGX Spark GB10/SM121
- **Quality:** parity with the BF16 baseline on a 200-question eval suite
  (GSM8K-flex / HumanEval / IFEval / agentic coding) — full results below

This checkpoint is a self-quantization by r0b0tlab. It is an independent
community artifact and is not affiliated with, endorsed by, or produced by
Ornith AI or NVIDIA.

## Quantization details

| Item | Value |
|---|---|
| Tool | NVIDIA Model Optimizer, dev tree @ `913f5e224` (`examples/hf_ptq` recipe flow) |
| Format | NVFP4 W4A16 weight-only: 4-bit E2M1 weights, block size 16, E4M3 per-block scales, scalar FP32 per-tensor scale |
| Scope | All `Linear` targets (routed experts, shared experts, dense attention projections, GDN linear-attention projections); lm_head, embeddings, routers, conv1d, MTP left in BF16 |
| Recipe | `w4_nvfp4` preset (max calibration) + `default_disabled_quantizers` + `kv_fp8_cast` |
| Calibration | `cnn_nemotron_v2_mix` (CNN/DailyMail + NVIDIA Nemotron Post-Training Dataset v2 mix), 1024 samples × 1024 tokens, batch 16 |
| KV cache | FP8-E4M3 with constant amax (cast; NVFP4 KV unavailable on aarch64) |
| MTP head | 785 BF16 tensors re-attached verbatim from the source checkpoint (byte-identical, SHA-256 audited) |

## Evaluation — 200-question suite (greedy, thinking disabled)

Fixed scorer (final-numeric-answer extraction). n = 80/40/40/20 per family;
±5–11 pp noise bands apply at these sample sizes.

| Family | BF16 baseline | This model (base AR) | This model + MTP |
|---|---|---|---|
| GSM8K (flex) | 76.25% (61/80) | 76.25% (61/80) | **78.75% (63/80)** |
| HumanEval pass@1 | 92.50% (37/40) | 92.50% (37/40) | **95.00% (38/40)** |
| IFEval (strict subset) | 90.00% (36/40) | 90.00% (36/40) | 90.00% (36/40) |
| Agentic coding | 70.00% (14/20) | 75.00% (15/20) | **85.00% (17/20)** |

**No measurable quality loss.** Speculative decoding with the BF16 MTP head
(K=1 draft): mean accept length 1.737, accept rate 0.737 over 964 decode
batches. Decode throughput on DGX Spark (GB10): 63–77 tok/s single-request
with CUDA graphs (vs 38.6 tok/s for an experts-only variant).

Per-tensor NVFP4 reconstruction cosines vs BF16: experts 0.9956–0.9973;
dense/GDN projections 0.967–0.986 (expected for plain max calibration on
outlier-rich projections; no end-to-end effect observed on this suite).

## Serving with SGLang

Validated command line (SGLang `0.5.6.post3.dev9218+g5a7b26c63`, FlashInfer
0.6.17, torch 2.13.0+cu130):

```bash
python -m sglang.launch_server \
  --model-path r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 \
  --served-model-name Ornith-1.5-35B-A3B \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend marlin \
  --context-length 32768 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --port 8000
```

Thinking-off protocol: pass `chat_template_kwargs: {"enable_thinking": false}`
in chat completion requests.

A ready-to-run Docker image with a pre-warmed FlashInfer JIT cache and a
one-command compose file is available at
[github.com/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang](https://github.com/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang).

### Known runtime notes (all root-caused during bring-up)

- **Warm the FlashInfer JIT cache before first boot.** First use of the
  dense FP4 GEMM modules JIT-compiles ~18 CUTLASS kernels; doing this with
  the model already resident can OOM smaller-memory systems. The provided
  container ships the cache pre-warmed.
- `--moe-runner-backend auto` may resolve to `flashinfer_trtllm`, which is
  not implemented for NVFP4 MoE in this SGLang version — pin `marlin` (as
  above).
- `FROZEN_KV_MTP` is not implemented for the qwen3_5 MTP head; use
  `--speculative-algorithm EAGLE` (the draft architecture is auto-remapped
  to `Qwen3_5ForCausalLMMTP`).

## How to verify

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.5-35B-A3B",
    "messages": [{"role": "user", "content": "What is 19*23? Answer with just the number."}],
    "temperature": 0, "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
# expect "437"
```

Reproduction scripts (200-question suite, scorer, audits) live in the GitHub
repo above; the full evidence trail (serve logs, quant logs, audit JSON) is
described there.

## Notes and limitations

- 200-question in-house suite; not a substitute for full public benchmarks.
- Evaluated thinking-off, greedy decoding, 32K context window.
- Dense/GDN per-tensor cosines are below the expert band (see above);
  out-of-distribution sensitivity beyond this suite was not probed.
- Throughput numbers are single-node DGX Spark (GB10/SM121, 121 GB unified);
  other Blackwell parts were not tested.

## Credits and attribution

- **Base model:** [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
  by Ornith AI (MIT license).
- **Quantization:** [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer)
  (dev @ `913f5e224`) — `w4_nvfp4` / `kv_fp8_cast` recipe units.
- **Calibration data:** [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
  (see the CNN/DailyMail papers) and
  [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)
  by NVIDIA.
- **Inference engine:** [SGLang](https://github.com/sgl-project/sglang)
  (`0.5.6.post3.dev9218+g5a7b26c63`) with [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
  0.6.17 CUTLASS FP4 GEMM kernels and [Marlin](https://github.com/IST-DASLab/marlin)
  W4A16 MoE kernels.
- **Hardware:** NVIDIA DGX Spark (GB10, SM 12.1).

## License

The base model is MIT-licensed; this derived quantization is distributed
under MIT as well. Calibration datasets remain under their own licenses.

## BibTeX

```bibtex
@misc{ornith15nvfp4w4a16,
  author  = {r0b0tlab},
  title   = {Ornith-1.5-35B-A3B-NVFP4-W4A16: NVFP4 weight-only quantization with BF16 MTP head},
  year    = {2026},
  url     = {https://huggingface.co/r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16},
  note    = {Quantized from ornith-ai/Ornith-1.5-35B-A3B with NVIDIA ModelOpt}
}
```
