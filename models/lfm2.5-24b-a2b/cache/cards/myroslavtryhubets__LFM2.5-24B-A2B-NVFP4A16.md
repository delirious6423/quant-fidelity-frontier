---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
- nvfp4
- compressed-tensors
- quantized
- moe
- lfm2.5
- liquid
---

# LFM2.5-24B-A2B, NVFP4 weights-only (W4A16)

An **unofficial, personal** 4-bit quantization of [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
(revision `1769516ac588c4c558089c94f27d96b0dc67b183`) that fits and runs on a single **16 GB consumer GPU**
(tested: RTX 5060 Ti 16 GB) with vLLM. It is not a Liquid AI release.

> **Notice of changes (License, section 4(b)):** the weights of this repository were modified by quantizing the
> linear layers to NVFP4 (see below). `config.json` carries a `quantization_config`; no other behavior was changed.
> Original model, copyright and attribution belong to Liquid AI. The original LFM Open License v1.0 is included as `LICENSE`
> and applies to this derivative.
>
> **License limit:** commercial use is licensed only for legal entities below US$10M annual revenue
> (including controlled/controlling entities); see `LICENSE`, section 5. Check it before any use at work.

## What is in here
| | |
|---|---|
| Format | compressed-tensors `nvfp4-pack-quantized`, **weights only** (`input_activations: null`) |
| Quantized | all expert `w1/w2/w3` (38 MoE layers x 64 experts), the 2 dense-layer MLPs, attention `q/k/v/out_proj`, conv `in_proj/out_proj` (7,402 linear layers) |
| Kept in bf16 | embeddings (tied `lm_head`), norms, q/k layernorms, router `gate`, `conv.conv`; `expert_bias` kept fp32 |
| Block scheme | 4-bit FP4 (e2m1) values, fp8 (e4m3) scale per 16 weights, one fp32 global scale per tensor (shared across fused groups q/k/v and w1/w3) |
| Rounding | per-block MSE search over 21 scale candidates, self-written quantizer (`scripts/quantize_lfm25.py`) |
| Size | 12.94 GiB on disk (bf16 original: 44.7 GiB) |
| Weight relative error | ‖W − Wq‖ / ‖W‖ = 0.081 for every layer type (`quant_stats.json`) |

## Run it (vLLM)
Tested with vLLM 0.30.0 on an RTX 5060 Ti 16 GB. vLLM uses the **Marlin weight-only NVFP4** path here (the GPU's FP4
tensor cores are not used for this checkpoint, so this saves memory, not compute). The weights take 12.95 GiB, so on a
16 GB card the KV cache is tiny and the flags below matter.

**Recommended on 16 GB (CUDA graphs + fp8 KV cache, ≈113k tokens of context):**
```bash
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864 \
vllm serve ./LFM2.5-24B-A2B-NVFP4A16 --served-model-name lfm \
  --kv-cache-dtype fp8 --max-model-len 113000 \
  --kv-cache-memory-bytes 1202590000 --gpu-memory-utilization 0.96 \
  --max-num-seqs 8 --cudagraph-capture-sizes 1 2 4 8 \
  --enable-auto-tool-choice --tool-call-parser lfm2 --reasoning-parser deepseek_r1
```
Why these flags (all measured on that card, not guaranteed elsewhere):
- `--cudagraph-capture-sizes 1 2 4 8` (no `--enforce-eager`): ≈160 tok/s single-stream decode at 1-4k context versus ≈40 tok/s eager.
- `--kv-cache-dtype fp8` halves the KV cache (≈10 KB/token). It only starts if the FlashInfer workspace is shrunk with
  `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864` (the 394 MiB default ran out of memory at every utilization ≥0.93 we tried).
- `--kv-cache-memory-bytes` sets the KV size explicitly (1.12 GiB ≈ 117k fp8 tokens). With only `--gpu-memory-utilization 0.96`, a
  server restarted with a warm `torch.compile` cache sized the KV cache ≈0.3 GiB too large and died with CUDA OOM on the first
  real request; the explicit size fixed that (cold and warm starts tested with a 106k-token prompt plus 8×6k and 8×20k concurrent).
- Parsers: `lfm2` parsed the model's `<|tool_call_start|>[fn(args)]<|tool_call_end|>` calls in our tests (120/120 calls, below).

Measured on that setup: GPU memory in use ≈15.4-15.6 GiB; decode ≈163 tok/s at 1k context, ≈144 at 32k, ≈125 at 64k, ≈108 at 100k;
prefill ≈4-7k tok/s (TTFT ≈24 s for a 100k prompt); 8 concurrent streams ≈380 tok/s summed decode.
Other measured options: fp8 KV + `--enforce-eager` at utilization 0.95 reaches the full 131072 context (≈40 tok/s); bf16 KV
gives ≈60k context with graphs (≈82k eager). CPU weight offload did not help (frees <0.5 GiB: the MoE experts, 98% of the
weights, are not offloaded; offloading all layers cost 3-4x decode speed).

**Minimal conservative command (what the first version of this card used):**
```bash
vllm serve ./LFM2.5-24B-A2B-NVFP4A16 --served-model-name lfm \
  --max-model-len 4096 --gpu-memory-utilization 0.93 --enforce-eager
```
That gives ≈42 tok/s per stream. *Correction:* an earlier version of this card said CUDA graphs run out of memory on 16 GB.
That was true only for those settings (0.93 with the default capture sizes); with the smaller capture set above they fit.

The model is a **reasoning model**: it always thinks first and the answer follows `</think>`. Use `max_tokens` of at least 8192
(one thinking turn can use ≈4k tokens; with a smaller cap answers are cut off). The base model's recommended sampling is
`temperature 0.1`, `top_k 50`, `repetition_penalty 1.05` (these are in `generation_config.json`; vLLM applies them by default);
the base model's own benchmarks used temperature 0.6.

## Quality: what we measured and what we did not
All numbers are our own small subsets with one sample per item; the base model's published figures use different caps and
sampling, so they are **not** a like-for-like comparison.

**Accuracy (T=0.6, top_p 0.95, top_k 50, repetition_penalty 1.05, `max_tokens=11000`, zero-shot, answer in `\boxed{}`):**
| Set (ours) | Accuracy (95% bootstrap CI) | Accuracy on items that finished thinking | Cut off at the token cap | Base model, published |
|---|---|---|---|---|
| MATH-500, first 150 problems | 92.7% (88.0-96.7) | 99.3% (139/140) | 6.7% (10 of 150) | 96.04 (full set, 32k output cap) |
| GSM8K, first 250 test problems | 95.6% (92.8-98.0) | 96.8% (239/247) | 1.2% (3 of 250) | not reported |

Items cut off before `</think>` count as wrong, which explains most of the MATH-500 gap. Our answer matcher is a simple
normalizer (no sympy), tuned once after looking at mismatches; one MATH item (#96) was hand-checked as equivalent but scored
wrong (93.3% if counted). Raw numbers: `eval_results.csv`, scripts: `scripts/eval_*.py`.

**Reference check against an 8-bit build (new).** We converted the original bf16 weights to a Q8_0 GGUF (near-lossless; ran in
llama.cpp with the MoE experts on CPU, same sampling and per-item seeds) and ran the first 50 MATH-500 + first 100 GSM8K problems:
| (same 150 problems) | accuracy (95% CI) |
|---|---|
| Q8_0 GGUF (llama.cpp) | 94.7% (90.7-98.0) |
| this NVFP4A16 checkpoint (vLLM, bf16 KV) | 96.0% (92.7-98.7) |

Difference +1.3 points (CI -2.0 to +4.7, McNemar p=0.69). The MATH part was 94% for both. So on this subset we saw **no measurable
loss from the 4-bit weights**; with n=150 the interval is wide (about ±3-4 points), so a loss of up to ≈3 points is not
excluded. The two runs use different engines, so the comparison is indicative, not paired. This is not a bf16 run.

**fp8 versus bf16 KV cache (same weights; vLLM, CUDA graphs):**
| test | bf16 KV | fp8 KV |
|---|---|---|
| MATH-500 first 150 + GSM8K first 250 | 95.5% | 94.5% (paired diff -1.0 pt, CI -2.8 to +0.7, p=0.42; bf16 eager vs graphs differ by the same amount) |
| HumanEval pass@1, 164 tasks x 2 samples | 96.3% | 97.3% (diff +0.9 pt, CI -0.9 to +2.7, p=0.51) |
| function calling, 120 prompts / 6 tools | 120/120 | 120/120 |
| own agent loop (context cap 45k, planted fact recalled at the end) | 4/4 valid runs | 4/5 (the miss was an answer cut off at the thinking limit) |

We found no difference between fp8 and bf16 KV on these tests (differences up to ≈1-3 points cannot be excluded).

**Long context (fp8 KV; synthetic task: 8 key→value notes scattered through Python source, answer as JSON):** equal to bf16 KV up to
72k tokens among finished answers; **at 96k and 112k only ≈54% and ≈42% of the values were correct** (bf16 KV cannot hold these
lengths on this card, so we cannot say whether fp8 or the model's own long-context limit is responsible). For agent work we
recommend keeping the live context below ≈70k even though the server accepts ≈113k. Our two-hop variant of the test was badly
designed and is not reported.

**Tool calling:** 120 prompts with 6 tools through vLLM's `lfm2` parser: 120/120 parsed as exactly one correct call (fp8 and bf16
KV). Earlier sanity check without the parser (20 prompts): 10/10 JSON extractions valid; of 10 tool-call prompts, 9 parsed as JSON
(3 in code fences) and 1 used the native `<|tool_call_start|>[...]<|tool_call_end|>` format.

**Not tested:** GPQA, IFEval, multilingual quality, the full 400-problem set against the 8-bit reference, long-context
reliability above ≈72k against a bf16 baseline, and any other quantization.

## Reproduce
`scripts/quantize_lfm25.py` (CPU, streams one shard at a time), `scripts/validate_lfm25.py` (dequantization check),
`scripts/eval_gen.py` / `scripts/eval_score.py` (the evaluation above). The first attempt with llm-compressor's
`memoryless_mse` observer on CPU did not finish one shard in 35 minutes, which is why the quantizer is self-written.

## Attribution
Base model: LFM2.5-24B-A2B by Liquid AI. See the base model card for architecture, intended use and limitations:
https://huggingface.co/LiquidAI/LFM2.5-24B-A2B
